"""English-language Streamlit interface for the independent simulated project."""
import hashlib
import io
import json
import os
import zipfile
from importlib.metadata import version

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from simulation import simulate, TRAITS, DEFAULT_COVARIATES, LAYERS
from planner import parse_local, parse_api, EXAMPLES, MODULES, PROMPT
from workflow import execute, build_graph, WORKFLOW_DOT

st.set_page_config(page_title='Trait & Multi-omics',page_icon='🧬',layout='wide')
st.markdown('''<style>
.stApp {background:#f7f9fc;}
h1,h2,h3 {color:#183854;}
div[data-testid="stMetric"] {background:white;border:1px solid #dce6ef;border-radius:12px;padding:14px;}
div[data-testid="stSidebar"] {background:#edf3f8;}
</style>''',unsafe_allow_html=True)
st.title('🧬 Trait & Multi-omics')
st.write('Ask a question. Review the plan. Explore molecular and genetic evidence.')
st.warning('SIMULATED DATA ONLY · All participants, molecules, genetic studies, links, and GO-like annotations are fictional. Results are a software demonstration.')


@st.cache_data(show_spinner=False)
def load_data(seed):
    return simulate(seed=seed)


@st.cache_data(show_spinner=False)
def workflow_mermaid():
    return build_graph().get_graph().draw_mermaid()


def show_table(frame, **kwargs):
    formatting={}
    for column in frame:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if column.startswith('PP_H') or column=='SNP_PP_H4':
            formatting[column]=st.column_config.NumberColumn(format='%.3f')
        elif column in ['p12','p_x','p_y'] or column.endswith(('p_value','q_value')):
            formatting[column]=st.column_config.NumberColumn(format='%.2e')
        elif pd.api.types.is_float_dtype(frame[column]):
            formatting[column]=st.column_config.NumberColumn(format='%.3f')
    st.dataframe(frame,column_config=formatting,**kwargs)


def zip_results(output, data, plan):
    buffer=io.BytesIO()
    versions={p:version(p) for p in ['langgraph','streamlit','numpy','scipy','statsmodels','pandas','openai']}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,table in output['tables'].items():
            archive.writestr(f'{name}.csv',table.to_csv(index=False))
        archive.writestr('report.md',output['report'])
        archive.writestr('analysis_plan.json',json.dumps(plan,indent=2))
        archive.writestr('provenance.json',json.dumps(dict(data=data['metadata'],software=versions,
            module_statuses=output['statuses'],trace=output['trace']),indent=2))
        archive.writestr('workflow.mmd',workflow_mermaid())
    return buffer.getvalue()


with st.sidebar:
    st.header('Demo settings')
    seed=int(st.number_input('Simulation seed',min_value=1,max_value=100000,value=2026,step=1))
    covariates=st.multiselect('Default adjustment covariates',DEFAULT_COVARIATES,default=DEFAULT_COVARIATES)
    fdr=st.number_input('BH-FDR threshold',min_value=.0001,max_value=.2,value=.05,step=.01,format='%.4f')
    modules=st.multiselect('Default analysis modules',MODULES,default=MODULES)
    st.caption('Explicit requests can change the plan. Review the resolved settings before running.')
    with st.expander('Try missing-data scenarios'):
        missing_methylation=st.checkbox('Hide DNA methylation dataset')
        missing_genetics=st.checkbox('Hide all genetic resources')
    st.header('Request interpretation')
    mode=st.radio('Mode',['Guided demo (no API)','OpenAI API'])
    key=model=''
    allow=False
    if mode=='OpenAI API':
        model=st.text_input('Model ID',value=os.getenv('OPENAI_MODEL',''),placeholder='A model available to your API project')
        key=st.text_input('API key (session only)',type='password') or os.getenv('OPENAI_API_KEY','')
        st.caption('Only your question, supported fields, and default settings are sent. No matrices, IDs, results, or genetic statistics are sent. Do not put individual information in the question.')
        allow=st.checkbox('Allow sending this question and configuration to OpenAI')
    else:
        st.caption('A bounded local parser recognizes SBP/HbA1c, supported adjustment clauses, and module names. It is not an LLM. For broader wording, use API mode.')
    st.caption('Supported traits: SBP (mmHg), HbA1c (percentage points). Independent participants; quantitative linear models only.')

data=load_data(seed)
if missing_methylation:
    data['methylation']=None
if missing_genetics:
    data['genetics']={}
fingerprint=hashlib.sha256(json.dumps([seed,covariates,fdr,modules,missing_methylation,missing_genetics,mode,model,'v1']).encode()).hexdigest()
if st.session_state.get('fingerprint')!=fingerprint:
    st.session_state.fingerprint=fingerprint
    for item in ['pending','output','question']:
        st.session_state.pop(item,None)

with st.expander('What is in this demo?'):
    st.write('600 independent synthetic participants · 60 proteins · 60 expression features · 24 miRNAs · 120 CpGs.')
    st.write('Planted associations allow positive and negative protein effects. Constant and missing features are included. Genetic regions illustrate shared signals, distinct signals, weak instruments, and unavailable statistics.')
    st.caption('Omics matrices use the same participant IDs. Expression values are simulated normalized signals; methylation beta values are converted to M-values before analysis. Genetic studies are generated separately from this cohort.')
    show_table(data['phenotype'].head(),width='stretch')
    st.json(data['metadata'])

st.subheader('Start with a question')
cols=st.columns(2)
if cols[0].button('Explore SBP · all modules',width='stretch'):
    st.session_state.example_question=EXAMPLES[0]
if cols[1].button('Explore HbA1c · all modules',width='stretch'):
    st.session_state.example_question=EXAMPLES[1]
with st.expander('Example questions and supported requests'):
    for example in EXAMPLES:
        st.code(example,language=None)
    st.write('Default modules come from the sidebar. Guided mode adds explicitly named modules; it does not interpret exclusions. Use sidebar controls for a smaller workflow. Subgroups, survival models, and new traits require further development.')

question=st.chat_input('Find proteins associated with SBP and integrate all multi-omics evidence, MR, colocalization, and GO.')
question=question or st.session_state.pop('example_question',None)
if question:
    st.session_state.pop('pending',None)
    st.session_state.pop('output',None)
    st.session_state.question=question
    try:
        if mode=='OpenAI API':
            if not allow:
                raise ValueError('Confirm permission to send the question and configuration to OpenAI.')
            with st.spinner('Interpreting your request…'):
                plan=parse_api(question,covariates,modules,fdr,model,key)
        else:
            plan=parse_local(question,covariates,modules,fdr)
        plan.update(request=question,interpretation_mode=mode)
        st.session_state.pending=plan
    except ValueError as error:
        st.error(f'Could not create a plan: {error}')

plan=st.session_state.get('pending')
if plan:
    with st.chat_message('user'):
        st.write(st.session_state.question)
    with st.chat_message('assistant'):
        st.markdown(f"**Plan:** screen all proteins for association with **{plan['trait']}** ({TRAITS[plan['trait']]}), adjusting for **{', '.join(plan['covariates']) or 'none'}**.")
        st.write(f"Then run: {', '.join(plan['modules']) or 'association only'}. BH-FDR < {plan['fdr']}.")
        st.caption('Trait ~ standardized feature + covariates; HC3 robust standard errors. Complete cases per test; minimum 50 samples; maximum 20% feature missingness. No imputation. site is categorical.')
        st.caption(plan['explanation'])
        if st.button('Run this analysis',type='primary'):
            try:
                with st.spinner('LangGraph is running the selected analysis workflow…'):
                    st.session_state.output=execute(data,plan)
            except Exception as error:
                st.error(f'Workflow could not be completed: {type(error).__name__}: {error}')

output=st.session_state.get('output')
if output:
    tables=output['tables']
    protein_results=tables['proteins']
    c=st.columns(4)
    c[0].metric('Proteins tested',output['summary']['proteins_tested'])
    c[1].metric('Protein hits',output['summary']['protein_hits'])
    mr=tables.get('mr',pd.DataFrame())
    coloc=tables.get('coloc',pd.DataFrame())
    c[2].metric('MR estimates',int(mr.status.eq('estimated').sum()) if not mr.empty else 'Not assessed')
    c[3].metric('Regions assessed',int(coloc.status.eq('estimated').sum()) if not coloc.empty else 'Not assessed')
    tabs=st.tabs(['Protein results','Multi-omics evidence','MR','Colocalization','GO enrichment','Workflow & downloads'])
    with tabs[0]:
        st.subheader(f"Protein associations with {plan['trait']}")
        st.caption('Effect: trait units per 1 SD higher protein; association is not evidence of causation.')
        evidence=tables['evidence_summary']
        leading=['protein','beta','ci_low','ci_high','q_value','significant']
        show_table(evidence[leading+[c for c in evidence if c not in leading]],hide_index=True,width='stretch')
        tested=protein_results.loc[protein_results.status.eq('tested')].copy()
        tested['minus_log10_q']=-np.log10(tested.q_value.clip(lower=1e-300))
        tested['Association']=np.where(tested.significant,'Meets FDR threshold','Above threshold')
        fig=px.scatter(tested,x='beta',y='minus_log10_q',color='Association',hover_name='feature',
            hover_data=['n','ci_low','ci_high','p_value','q_value'],
            labels={'beta':f"{plan['trait']} effect per protein SD ({TRAITS[plan['trait']]})",'minus_log10_q':'−log10(BH-FDR)'},
            color_discrete_map={'Meets FDR threshold':'#167d9a','Above threshold':'#b6c4cf'})
        fig.add_hline(y=-np.log10(plan['fdr']),line_dash='dash')
        st.plotly_chart(fig,width='stretch')
        if output['candidates']:
            selected=st.selectbox('Inspect a protein',output['candidates'])
            view=pd.concat([data['proteins'][selected],data['phenotype'][plan['trait']]],axis=1).dropna()
            st.plotly_chart(px.scatter(view,x=selected,y=plan['trait'],opacity=.45,title='Unadjusted observations — the table above contains adjusted estimates'),width='stretch')
    with tabs[1]:
        layer=st.selectbox('Evidence layer',LAYERS)
        status=output['statuses'].get(layer,{})
        st.info(f"{status.get('status','not assessed')}: {status.get('reason','')}")
        if layer in tables:
            st.write('All feature–trait associations, corrected across this molecular layer')
            show_table(tables[layer],hide_index=True,width='stretch')
            st.write('Mapped evidence for candidate proteins')
            st.caption('trait_* columns: feature–trait tests. protein_link_* columns: adjusted protein–feature tests, corrected across ALL registered links in this layer before candidate filtering. Positional/target annotations are fictional hypotheses, not demonstrated regulation.')
            show_table(tables[f'{layer}_links'],hide_index=True,width='stretch')
    with tabs[2]:
        status=output['statuses'].get('mr',{})
        st.info(f"{status.get('status','not assessed')}: {status.get('reason','')}")
        if not mr.empty:
            leading=[c for c in ['protein','beta','ci_low','ci_high','q_value','status'] if c in mr]
            show_table(mr[leading+[c for c in mr if c not in leading]],hide_index=True,width='stretch')
            estimated=mr.loc[mr.status.eq('estimated')]
            if not estimated.empty:
                fig=px.scatter(estimated,x='beta',y='protein',error_x=estimated.ci_high-estimated.beta,
                    error_x_minus=estimated.beta-estimated.ci_low,hover_data=['n_snps','p_value','q_value'],
                    labels={'beta':f"SIMULATED {plan['trait']} change ({TRAITS[plan['trait']]}) per protein SD"},
                    title='SIMULATED MR — not evidence about real disease biology')
                fig.add_vline(x=0,line_dash='dash')
                st.plotly_chart(fig,width='stretch')
            st.caption('cis-pQTL p ≤ 5e−8, F ≥ 10; LD pruning r² ≤ 0.001 using the simulation LD matrix; drop palindromic alleles and align effects. Wald ratio or fixed-effect IVW. Candidate-family MR FDR is exploratory after screening. No MR-Egger or complete pleiotropy validation.')
            with st.expander('Instrument checks, including excluded SNPs'):
                show_table(tables['instrument_qc'],hide_index=True,width='stretch')
    with tabs[3]:
        status=output['statuses'].get('coloc',{})
        st.info(f"{status.get('status','not assessed')}: {status.get('reason','')}")
        if not coloc.empty:
            leading=[c for c in ['protein','PP_H3','PP_H4','n_snps','status'] if c in coloc]
            show_table(coloc[leading+[c for c in coloc if c not in leading]],hide_index=True,width='stretch')
            st.caption('Python implementation of Wakefield ABF H0–H4 enumeration, not a call to R coloc. One causal variant per trait per region. H3: different causal variants; H4: shared causal variant. Neither proves that protein mediates the trait.')
            valid=coloc.loc[coloc.status.eq('estimated'),'protein'].tolist()
            if valid:
                chosen=st.selectbox('Inspect a genetic region',valid)
                dense=tables['coloc_snps'].loc[lambda x:x.protein.eq(chosen)]
                values=dense.melt(id_vars=['snp','position'],value_vars=['p_x','p_y'],var_name='Study',value_name='p')
                values['minus_log10_p']=-np.log10(values.p.clip(lower=1e-300))
                values['Study']=values.Study.map({'p_x':'Simulated pQTL','p_y':f"Simulated {plan['trait']} GWAS"})
                st.plotly_chart(px.scatter(values,x='position',y='minus_log10_p',color='Study',hover_name='snp',
                    title=f'{chosen} · full simulated region',labels={'position':'Fictional genomic position','minus_log10_p':'−log10(p)'}),width='stretch')
                st.write('Sensitivity to the shared-variant prior (p12)')
                priors=tables['coloc_priors'].loc[lambda x:x.protein.eq(chosen)]
                show_table(priors[['p12','PP_H3','PP_H4','PP_H0','PP_H1','PP_H2']],hide_index=True)
    with tabs[4]:
        status=output['statuses'].get('go',{})
        st.info(f"{status.get('status','not assessed')}: {status.get('reason','')}")
        st.warning('GO-style demonstration with FICTIONAL SIM_GO gene sets. No real Gene Ontology annotations or biological enrichment claims.')
        if 'go' in tables:
            go=tables['go']
            show_table(go,hide_index=True,width='stretch')
            if not go.empty:
                fig=px.bar(go.sort_values('q_value').head(8),x='overlap',y='name',color='q_value',orientation='h',
                    hover_data=['p_value','background_count','hit_count'],title='Overlap with simulated gene sets')
                st.plotly_chart(fig,width='stretch')
            st.caption('Hypergeometric over-representation; BH across all tested sets. Background: successfully tested, uniquely mapped protein genes. This version uses the protein discovery set only.')
    with tabs[5]:
        st.graphviz_chart(WORKFLOW_DOT,width='stretch')
        st.subheader('Execution log')
        show_table(pd.DataFrame(output['trace']),hide_index=True,width='stretch')
        with st.expander('LangGraph structure'):
            st.code(workflow_mermaid(),language='mermaid')
        with st.expander('Analysis plan and provenance'):
            st.json(plan)
            st.json(data['metadata'])
        st.download_button('Download complete analysis bundle (ZIP)',zip_results(output,data,plan),
            file_name=f"SIMULATED_{plan['trait']}_analysis.zip",mime='application/zip')
        st.download_button('Download report',output['report'],file_name='SIMULATED_report.md',mime='text/markdown')
        st.download_button('Download evidence summary CSV',tables['evidence_summary'].to_csv(index=False).encode('utf-8-sig'),
            file_name='SIMULATED_evidence_summary.csv',mime='text/csv')
        with st.expander('Read report'):
            st.markdown(output['report'])

with st.expander('How it works and current limits'):
    st.write('LangGraph runs real Python analysis functions. Expression, miRNA, and methylation nodes run as parallel branches; MR and colocalization are separate branches. Missing modules retain explicit status. Reports are built from result tables, without an LLM inventing numbers.')
    st.write('This version supports only its own simulated data and two continuous traits. It does not fit survival, binary, family, or repeated-measure models, perform mediation, or use real regulatory/GO databases. A failed request clears old results; changing settings requires a new plan.')
    st.code(PROMPT,language='text')
    st.markdown('[LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/workflows-agents) · '
                '[coloc assumptions](https://chr1swallace.github.io/coloc/articles/a03_enumeration.html) · '
                '[GO background guidance](https://www.geneontology.org/docs/go-enrichment-analysis/)')
