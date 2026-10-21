"""Export reproducible synthetic inputs and a complete local example analysis."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from simulation import simulate, DEFAULT_COVARIATES
from planner import parse_local, MODULES
from workflow import execute, build_graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--trait',choices=['SBP','HbA1c'],default='SBP')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs'/'example_SBP')
    parser.add_argument('--export-inputs',action='store_true')
    args=parser.parse_args()
    data=simulate(seed=args.seed)
    question=f'Find proteins associated with {args.trait} and integrate all analyses.'
    plan=parse_local(question,DEFAULT_COVARIATES,MODULES,.05)
    plan.update(request=question,interpretation_mode='Guided local CLI')
    result=execute(data,plan)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,table in result['tables'].items():
        table.to_csv(args.output/f'{name}.csv',index=False)
    (args.output/'report.md').write_text(result['report'],encoding='utf-8')
    provenance=dict(plan=plan,metadata=data['metadata'],statuses=result['statuses'],trace=result['trace'],
        software={name:version(name) for name in ['numpy','pandas','scipy','statsmodels','langgraph','streamlit']})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (args.output/'workflow.mmd').write_text(build_graph().get_graph().draw_mermaid(),encoding='utf-8')
    if args.export_inputs:
        target=Path(__file__).parent/'demo_data'
        target.mkdir(exist_ok=True)
        for key in ['phenotype','proteins','expression','mirna','methylation']:
            data[key].to_csv(target/f'{key}.csv')
        data['links'].to_csv(target/'synthetic_molecular_links.csv',index=False)
        data['ontology'].to_csv(target/'FICTIONAL_GO_annotations.csv',index=False)
        pqtl,outcomes=[],[]
        scenarios=[]
        for protein,region in data['genetics'].items():
            pqtl.append(region['pqtl'].assign(protein=protein))
            for trait,frame in region['outcomes'].items():
                outcomes.append(frame.assign(protein_region=protein,trait=trait))
            scenarios.append(dict(protein=protein,scenario=region['scenario']))
        pd.concat(pqtl,ignore_index=True).to_csv(target/'SIMULATED_pqtl.csv',index=False)
        pd.concat(outcomes,ignore_index=True).to_csv(target/'SIMULATED_trait_GWAS.csv',index=False)
        pd.DataFrame(scenarios).to_csv(target/'simulation_scenarios.csv',index=False)
        np.save(target/'simulated_region_LD.npy',next(iter(data['genetics'].values()))['ld'])
        (target/'metadata.json').write_text(json.dumps(data['metadata'],indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=result['summary'],output=str(args.output)),indent=2))


if __name__=='__main__':
    main()
