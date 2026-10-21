"""Scientific and workflow regressions, including missing-data and no-hit paths."""
import copy
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from simulation import simulate, DEFAULT_COVARIATES
from statistics_engine import (scan, harmonize_region, select_instruments, mr_estimate,
    coloc_abf, go_enrichment, molecular_evidence)
from planner import parse_local, parse_api, EXAMPLES, MODULES, Intent
from workflow import execute


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=simulate()
        cls.plan=parse_local(EXAMPLES[0],DEFAULT_COVARIATES,MODULES,.05)
        cls.output=execute(cls.data,cls.plan)

    def test_reproducible_simulation_and_matching_samples(self):
        other=simulate()
        pd.testing.assert_frame_equal(other['proteins'],self.data['proteins'])
        for layer in ['proteins','expression','mirna','methylation']:
            self.assertTrue(self.data[layer].index.equals(self.data['phenotype'].index))
        self.assertNotIn('SIM_P004',self.data['genetics'])

    def test_known_protein_signs_and_failed_features(self):
        table=self.output['tables']['proteins'].set_index('feature')
        self.assertTrue(table.loc['SIM_P001','significant'])
        self.assertGreater(table.loc['SIM_P001','beta'],0)
        self.assertTrue(table.loc['SIM_P002','significant'])
        self.assertLess(table.loc['SIM_P002','beta'],0)
        self.assertEqual(table.loc['SIM_P060','status'],'skipped: constant feature')
        self.assertIn('missingness',table.loc['SIM_P059','status'])
        self.assertEqual(table.status.eq('tested').sum(),58)

    def test_adjustment_removes_artificial_confounding(self):
        rng=np.random.default_rng(70)
        n=1000
        confounder=rng.normal(size=n)
        frame=pd.DataFrame({'X':confounder+rng.normal(0,.3,n)})
        phenotype=pd.DataFrame({'age':confounder,'SBP':2*confounder+rng.normal(0,.3,n)})
        unadjusted=scan(frame,phenotype,'SBP',[]).iloc[0]
        adjusted=scan(frame,phenotype,'SBP',['age']).iloc[0]
        self.assertGreater(unadjusted.beta,1.5)
        self.assertLess(abs(adjusted.beta),.2)

    def test_layer_family_is_not_restricted_to_selected_proteins(self):
        all_results,evidence=molecular_evidence(self.data,'expression',['SIM_P001'],self.plan)
        self.assertEqual(len(all_results),60)
        self.assertEqual(evidence.protein.tolist(),['SIM_P001'])
        all_results2,evidence2=molecular_evidence(self.data,'expression',['SIM_P001','SIM_P002'],self.plan)
        self.assertEqual(evidence.protein_link_q_value.iloc[0],
                         evidence2.loc[evidence2.protein.eq('SIM_P001'),'protein_link_q_value'].iloc[0])
        self.assertIn('trait_q_value',evidence)

    def test_harmonization_reverses_outcome_alleles(self):
        region=self.data['genetics']['SIM_P002']
        dense=harmonize_region(region,'SBP')
        self.assertTrue(dense.alignment.eq('flipped').all())
        self.assertLess(dense.loc[dense.snp.str.endswith('_040'),'beta_y'].iloc[0],0)
        qc=select_instruments(dense,region)
        estimate=mr_estimate(qc.loc[qc.decision.eq('eligible')])
        self.assertLess(estimate['beta'],0)

    def test_mr_recovers_known_effect(self):
        instruments=pd.DataFrame({'beta_x':[.2,.3,.4],'se_x':[.001]*3,
            'beta_y':[.1,.15,.2],'se_y':[.02]*3})
        estimate=mr_estimate(instruments)
        self.assertAlmostEqual(estimate['beta'],.5)
        self.assertEqual(estimate['n_snps'],3)
        self.assertAlmostEqual(estimate['Q'],0)
        single=mr_estimate(instruments.iloc[:1])
        self.assertAlmostEqual(single['beta'],.5)
        self.assertGreater(single['se'],0)

    def test_coloc_distinguishes_shared_and_distinct_signals(self):
        for protein, hypothesis in [('SIM_P001','PP_H4'),('SIM_P003','PP_H3')]:
            dense=harmonize_region(self.data['genetics'][protein],'SBP')
            pp,snp_pp=coloc_abf(dense.beta_x,dense.se_x,dense.beta_y,dense.se_y,15)
            self.assertGreater(pp[hypothesis],.99)
            self.assertAlmostEqual(sum(pp.values()),1)
            self.assertAlmostEqual(sum(snp_pp),1)
            self.assertTrue(np.isfinite(list(pp.values())).all())

    def test_coloc_uses_full_region_not_pruned_instruments(self):
        table=self.output['tables']['coloc'].set_index('protein')
        self.assertEqual(table.loc['SIM_P001','n_snps'],101)
        mr=self.output['tables']['mr'].set_index('protein')
        self.assertLess(mr.loc['SIM_P001','n_snps'],101)
        self.assertEqual(len(self.output['tables']['coloc_priors'].loc[lambda x:x.protein.eq('SIM_P001')]),3)

    def test_mr_no_tools_does_not_prevent_coloc(self):
        region=self.data['genetics']['SIM_P006']
        dense=harmonize_region(region,'SBP')
        qc=select_instruments(dense,region)
        self.assertFalse(qc.decision.eq('eligible').any())
        pp,_=coloc_abf(dense.beta_x,dense.se_x,dense.beta_y,dense.se_y,15)
        self.assertGreater(pp['PP_H2'],.8)

    def test_go_has_correct_background_and_zero_overlap_tests(self):
        tables=self.output['tables']
        self.assertEqual(len(tables['go_background']),58)
        self.assertNotIn('SIM_GENE_059',tables['go_background'].gene.tolist())
        self.assertEqual(len(tables['go']),8)
        self.assertTrue(tables['go'].loc[tables['go'].overlap.eq(0),'p_value'].eq(1).all())
        annotations=pd.DataFrame({'term':['A']*3+['B']*2,'name':['set A']*3+['set B']*2,
                                   'gene':['g1','g2','g3','g8','g9']})
        result=go_enrichment(['g1','g2','g2'],[f'g{i}' for i in range(1,11)],annotations).set_index('term')
        self.assertAlmostEqual(result.loc['A','p_value'],hypergeom.sf(1,10,3,2))
        self.assertEqual(result.loc['A','hit_count'],2)

    def test_missing_resources_are_explicit(self):
        data=dict(self.data,methylation=None,genetics={})
        result=execute(data,self.plan)
        self.assertEqual(result['statuses']['methylation']['status'],'skipped')
        self.assertTrue(result['tables']['mr'].status.eq('not assessed').all())
        self.assertTrue(result['tables']['coloc'].status.eq('not assessed').all())
        self.assertTrue(result['tables']['evidence_summary'].methylation_trait_hits.isna().all())

    def test_no_hits_gate_retains_full_results(self):
        plan=dict(self.plan,fdr=1e-100)
        result=execute(self.data,plan)
        self.assertEqual(result['candidates'],[])
        self.assertEqual(len(result['tables']['proteins']),60)
        self.assertNotIn('mr',result['tables'])
        self.assertEqual(result['statuses']['mr']['status'],'skipped')
        self.assertEqual(result['trace'][-1]['node'],'verify')

    def test_association_only_and_second_trait(self):
        plan=parse_local('Find proteins associated with HbA1c.',DEFAULT_COVARIATES,[],.05)
        result=execute(self.data,plan)
        self.assertGreater(result['summary']['protein_hits'],0)
        self.assertNotIn('mr',result['tables'])
        self.assertEqual(result['statuses']['expression']['status'],'skipped')

    def test_reject_real_or_correlated_data(self):
        for key in ['synthetic','independent_participants']:
            data=dict(self.data,metadata={**self.data['metadata'],key:False})
            with self.assertRaises(ValueError):
                execute(data,self.plan)

    def test_local_parser_covariates_ambiguity_and_unknowns(self):
        plan=parse_local(EXAMPLES[2],DEFAULT_COVARIATES,[],.05)
        self.assertEqual(plan['covariates'],['age','sex'])
        self.assertEqual(parse_local(EXAMPLES[1],[],[],.05)['trait'],'HbA1c')
        for question in ['Find proteins associated with blood pressure.',
            'Find proteins associated with SBP and HbA1c.',
            'Find proteins associated with SBP, adjusting for smoking.',
            'Find proteins associated with SBP in men only.',
            'Find proteins associated with mortality.']:
            with self.assertRaises(ValueError):
                parse_local(question,DEFAULT_COVARIATES,MODULES,.05)

    def test_api_payload_is_configuration_only(self):
        intent=Intent(action='ready',trait='SBP',covariates=['age'],modules=['mr'],explanation='Example')
        client=MagicMock()
        client.__enter__.return_value=client
        client.responses.parse.return_value.output_parsed=intent
        with patch('openai.OpenAI',return_value=client):
            result=parse_api('Find SBP proteins',['age'],['mr'],.05,'test-model','secret-do-not-log')
        kwargs=client.responses.parse.call_args.kwargs
        payload=json.loads(kwargs['input'][1]['content'])
        self.assertEqual(set(payload),{'question','supported_traits','available_covariates','default_covariates','default_modules'})
        self.assertFalse(kwargs['store'])
        self.assertNotIn('secret-do-not-log',str(kwargs))
        self.assertEqual(result['modules'],['mr'])

    def test_graph_parallel_branches_and_report_validation(self):
        nodes=[t['node'] for t in self.output['trace']]
        for node in ['expression','mirna','methylation','mr','coloc','go','verify']:
            self.assertIn(node,nodes)
        self.assertGreater(nodes.index('genetic_resources'),max(nodes.index(m) for m in ['expression','mirna','methylation']))
        self.assertGreater(nodes.index('go'),max(nodes.index('mr'),nodes.index('coloc')))
        self.assertIn('SIMULATED ONLY',self.output['report'])


if __name__=='__main__':
    unittest.main(verbosity=2)
