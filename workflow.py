"""Actual LangGraph fan-out/fan-in, conditional routing, local computations."""
import operator
from typing import Annotated, Any, TypedDict
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, START, END
from statistics_engine import (scan, molecular_evidence, harmonize_region, select_instruments,
    mr_estimate, coloc_abf, go_enrichment, correct_fdr)
from planner import validate_plan, Intent
from simulation import LAYERS, VERSION, TRAITS

WORKFLOW_DOT = '''digraph {
  graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.4];
  node [shape=box, style="rounded,filled", fillcolor="#eaf2fb", color="#7d9dbc", fontname="Arial", fontsize=12];
  request [label="Natural-language request → reviewed analysis plan"];
  screen [label="Protein–trait association"];
  gate [label="Protein hits?", shape=diamond, fillcolor="#fff2cf"];
  expression [label="Gene expression"];
  mirna [label="miRNA"];
  methylation [label="DNA methylation"];
  resources [label="Simulated pQTL + trait GWAS"];
  mr [label="MR\nEligible instruments only"];
  coloc [label="Colocalization\nFull regional statistics"];
  go [label="GO-style enrichment\nFictional annotations"];
  report [label="Evidence report + validation", fillcolor="#dcf1e6"];
  request -> screen -> gate;
  gate -> expression [label="yes"];
  gate -> mirna;
  gate -> methylation;
  gate -> report [label="no: keep full screen"];
  expression -> resources;
  mirna -> resources;
  methylation -> resources;
  resources -> mr;
  resources -> coloc;
  mr -> go;
  coloc -> go;
  go -> report;
}'''


class AnalysisState(TypedDict, total=False):
    data: Any
    plan: dict
    tables: Annotated[dict, lambda a,b: {**a, **b}]
    trace: Annotated[list, operator.add]
    statuses: Annotated[dict, lambda a,b: {**a, **b}]
    candidates: list[str]
    regions: Any
    evidence: Any
    summary: dict
    report: str


def update(node, status, reason, tables=None, **extra):
    return dict(trace=[dict(node=node,status=status,detail=reason)],
        statuses={node:dict(status=status,reason=reason)},tables=tables or {},**extra)


def validate(state):
    if not state['data']['metadata'].get('synthetic'):
        raise ValueError('This project accepts its own synthetic data only.')
    if not state['data']['metadata'].get('independent_participants'):
        raise ValueError('Related or repeated participants require a different model.')
    plan=state['plan']
    validate_plan(Intent(action='ready',trait=plan['trait'],covariates=plan['covariates'],
        modules=plan['modules'],explanation=plan['explanation']),plan['fdr'])
    return update('validate','completed','Synthetic cohort; unique IDs and independent participants; quantitative trait model.')


def protein_association(state):
    d,p=state['data'],state['plan']
    result=scan(d['proteins'],d['phenotype'],p['trait'],p['covariates'],p['fdr'])
    candidates=result.loc[result.significant,'feature'].tolist()
    return update('protein_association','completed',f'{len(candidates)} hits among {int(result.status.eq("tested").sum())} tested proteins.',
        {'proteins':result},candidates=candidates)


def layer_node(layer):
    def run(state):
        if layer not in state['plan']['modules']:
            return update(layer,'skipped','Not requested.')
        d=state['data']
        if layer not in d or d[layer] is None:
            return update(layer,'skipped','Molecular dataset unavailable; not assessed.')
        try:
            results, evidence=molecular_evidence(d,layer,state['candidates'],state['plan'])
            return update(layer,'completed',f'{int(results.status.eq("tested").sum())} features tested; {int(results.significant.sum())} trait hits; {len(evidence)} candidate links.',
                {layer:results,f'{layer}_links':evidence})
        except (ValueError,KeyError,np.linalg.LinAlgError) as error:
            return update(layer,'failed',f'Analysis failed: {error}')
    return run


def genetic_resources(state):
    if not any(m in state['plan']['modules'] for m in ['mr','coloc']):
        return update('genetic_resources','skipped','Genetic analyses not requested.',regions={})
    regions={}
    for protein in state['candidates']:
        resource=state['data']['genetics'].get(protein)
        if resource is not None:
            regions[protein]=resource
    return update('genetic_resources','completed',f'{len(regions)} candidate pQTL resources found; all synthetic.',regions=regions)


def mr_node(state):
    if 'mr' not in state['plan']['modules']:
        return update('mr','skipped','Not requested.')
    rows,qcs=[],[]
    for protein in state['candidates']:
        row=dict(protein=protein,status='not assessed',reason='',p_value=np.nan)
        region=state['regions'].get(protein)
        if region is None:
            row['reason']='pQTL resource unavailable'
        else:
            dense=harmonize_region(region,state['plan']['trait'])
            if dense.empty:
                row['reason']='Trait GWAS unavailable'
            else:
                qc=select_instruments(dense,region)
                qc.insert(0,'protein',protein)
                qcs.append(qc)
                instruments=qc.loc[qc.decision.eq('eligible')]
                if instruments.empty:
                    row['reason']='No eligible instruments after strength, allele, and simulated LD checks'
                else:
                    row.update(mr_estimate(instruments),reason='Synthetic independent samples; no causal claim for real biology')
        rows.append(row)
    result=correct_fdr(pd.DataFrame(rows),state['plan']['fdr'])
    result['source']='SIMULATED ONLY'
    return update('mr','completed',f'{int(result.status.eq("estimated").sum())} estimates; remaining candidates explicitly not assessed.',
        {'mr':result,'instrument_qc':pd.concat(qcs,ignore_index=True) if qcs else pd.DataFrame()})


def coloc_node(state):
    if 'coloc' not in state['plan']['modules']:
        return update('coloc','skipped','Not requested.')
    rows, sensitivities, snp_rows=[],[],[]
    sd_y=float(state['data']['phenotype'][state['plan']['trait']].std(ddof=1))
    for protein in state['candidates']:
        row=dict(protein=protein,status='not assessed',reason='',n_snps=0)
        region=state['regions'].get(protein)
        if region is None:
            row['reason']='pQTL resource unavailable'
        else:
            dense=harmonize_region(region,state['plan']['trait'])
            if dense.empty:
                row['reason']='Trait GWAS unavailable'
            else:
                dense=dense.loc[dense.status.eq('aligned')]
                if len(dense)<20:
                    row['reason']='Insufficient dense, aligned regional statistics'
                else:
                    for prior in [1e-6,1e-5,1e-4]:
                        pp,conditional=coloc_abf(dense.beta_x,dense.se_x,dense.beta_y,dense.se_y,sd_y,p12=prior)
                        sensitivities.append(dict(protein=protein,p12=prior,**pp))
                        if prior == 1e-5:
                            row.update(pp,n_snps=len(dense),status='estimated',
                                reason='Single causal variant per trait assumed; posterior is not disease-causality probability',
                                lead_snp_H4=dense.iloc[int(np.argmax(conditional))].snp)
                            snp_rows.append(dense.assign(protein=protein,SNP_PP_H4=conditional))
        rows.append(row)
    return update('coloc','completed',f'{sum(r["status"]=="estimated" for r in rows)} regional ABF estimates; no p-value filtering or MR clumping.',
        {'coloc':pd.DataFrame(rows),'coloc_priors':pd.DataFrame(sensitivities),
         'coloc_snps':pd.concat(snp_rows,ignore_index=True) if snp_rows else pd.DataFrame()})


def go_node(state):
    if 'go' not in state['plan']['modules']:
        return update('go','skipped','Not requested.')
    protein_results=state['tables']['proteins']
    tested=protein_results.loc[protein_results.status.eq('tested'),'feature']
    mapping=state['data']['links'].loc[state['data']['links'].layer.eq('expression')].set_index('protein').gene
    background=mapping.reindex(tested).dropna().unique().tolist()
    hits=mapping.reindex(state['candidates']).dropna().unique().tolist()
    result=go_enrichment(hits,background,state['data']['ontology'],state['plan']['fdr'])
    return update('go','completed',f'{len(result)} fictional gene sets tested; {len(background)} tested-protein genes in background.',
        {'go':result,'go_background':pd.DataFrame({'gene':background}),'go_foreground':pd.DataFrame({'gene':hits})})


def empty_candidates(state):
    modules=LAYERS+['genetic_resources','mr','coloc','go']
    return dict(statuses={m:dict(status='skipped',reason='No protein hits; downstream candidate workflow skipped.') for m in modules},
        trace=[dict(node='candidate_gate',status='skipped',detail='No protein hits. Full association table retained.')])


def report(state):
    p,t=state['plan'],state['tables']
    summary=t['proteins'].rename(columns={'feature':'protein'}).copy()
    for layer in LAYERS:
        links=t.get(f'{layer}_links')
        counts={} if links is None else links.loc[links.trait_significant].groupby('protein').feature.nunique().to_dict()
        summary[f'{layer}_trait_hits']=summary.protein.map(counts).fillna(0).astype(int) if links is not None else pd.NA
    for module in ['mr','coloc']:
        result=t.get(module)
        if result is not None:
            cols=['protein','status','q_value','beta'] if module=='mr' else ['protein','status','PP_H3','PP_H4']
            cols=[c for c in cols if c in result]
            summary=summary.merge(result[cols].rename(columns={c:f'{module}_{c}' for c in cols if c!='protein'}),on='protein',how='left')
    source='SIMULATED ONLY — fictional identifiers, molecular annotations and GO-like gene sets.'
    summary['source']=source
    tested=int(t['proteins'].status.eq('tested').sum())
    hits=len(state['candidates'])
    text=f'''# Multi-omics trait exploration — SIMULATED ONLY

Trait: {p['trait']} ({TRAITS[p['trait']]}). Model: {p['model']}.
Covariates: {', '.join(p['covariates']) or 'none'}. BH-FDR threshold: {p['fdr']}.

Tested {tested} proteins; {hits} association hits. See proteins.csv (feature identifier)
and evidence_summary.csv (protein identifier) for every estimate, CI, sample size and QC decision.

Layer trait hits and protein–feature associations are distinct evidence columns.
BH adjustment is separate for each full molecular layer and each full registered link family.
MR BH adjustment covers successfully estimated candidate proteins after screening;
it is an exploratory downstream family, not correction over all possible proteins.

MR effects are {TRAITS[p['trait']]} per SD higher protein in artificial independent studies.
Colocalization uses Wakefield ABFs, quantitative prior SD 0.15 times phenotype SD,
and one causal variant per trait per region. p1=p2=1e-4; p12=1e-5 with sensitivity
at 1e-6 and 1e-4. Full aligned regions, not selected MR instruments, are used.
No SuSiE or multiple-signal model is implemented. H4 supports a shared variant,
not a causal protein effect or a complete multi-omics mediation chain.

GO-style enrichment uses FICTIONAL gene sets (SIM_GO identifiers), NOT downloaded GO
annotations. The background contains successfully tested, mapped protein genes.
All tested sets enter BH correction, including sets with no overlapping hits.

The cohort and genetic summary statistics are separate artificial simulations;
agreement is deliberately illustrative and disagreement is possible. Simulated
LD is known from the generator. MR assumes valid instruments and independent studies;
pleiotropy, reverse causality, assay-binding artifacts and winner's curse are not resolved.

No LLM generates statistical numbers. This report is assembled from result tables.
Every result is for software demonstration, not a finding about real human biology.
'''
    for node, entry in sorted(state['statuses'].items()):
        text+=f'\n- {node}: {entry["status"]} — {entry["reason"]}'
    return update('report','completed','Report generated deterministically from result tables.',
        {'evidence_summary':summary},report=text,summary=dict(trait=p['trait'],proteins_tested=tested,
        protein_hits=hits,source=source,simulation_version=VERSION))


def verify(state):
    table=state['tables']['proteins']
    if int(table.significant.sum()) != state['summary']['protein_hits']:
        raise ValueError('Report validation failed: inconsistent hit count.')
    if set(state['candidates']) != set(table.loc[table.significant,'feature']):
        raise ValueError('Report validation failed: inconsistent candidates.')
    coloc=state['tables'].get('coloc')
    if coloc is not None and 'PP_H4' in coloc:
        valid=coloc.loc[coloc.status.eq('estimated'),[f'PP_H{i}' for i in range(5)]]
        if not np.isfinite(valid.to_numpy()).all() or not np.allclose(valid.sum(axis=1),1):
            raise ValueError('Report validation failed: invalid coloc posterior probabilities.')
    if 'SIMULATED ONLY' not in state['report']:
        raise ValueError('Report validation failed: missing simulation label.')
    return update('verify','completed','Hit counts, candidate identifiers, posterior normalization and simulation labels checked.')


def build_graph():
    graph=StateGraph(AnalysisState)
    for name,node in [('validate',validate),('protein_association',protein_association),
        ('empty_candidates',empty_candidates),('genetic_resources',genetic_resources),
        ('mr',mr_node),('coloc',coloc_node),('go',go_node),('report',report),('verify',verify)]:
        graph.add_node(name,node)
    for layer in LAYERS:
        graph.add_node(layer,layer_node(layer))
    graph.add_edge(START,'validate')
    graph.add_edge('validate','protein_association')
    graph.add_conditional_edges('protein_association',lambda s: 'omics' if s['candidates'] else 'empty',
        {'omics':'omics_start','empty':'empty_candidates'})
    graph.add_node('omics_start',lambda s: {})
    for layer in LAYERS:
        graph.add_edge('omics_start',layer)
    graph.add_edge(LAYERS,'genetic_resources')
    graph.add_edge('genetic_resources','mr')
    graph.add_edge('genetic_resources','coloc')
    graph.add_edge(['mr','coloc'],'go')
    graph.add_edge('go','report')
    graph.add_edge('empty_candidates','report')
    graph.add_edge('report','verify')
    graph.add_edge('verify',END)
    return graph.compile()


def execute(data, plan):
    from langsmith import tracing_context
    with tracing_context(enabled=False):
        return build_graph().invoke(dict(data=data,plan=plan,tables={},trace=[],statuses={}),
                                   config={'max_concurrency':3})
