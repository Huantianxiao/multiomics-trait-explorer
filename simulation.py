"""Reproducible artificial cohort, annotations, and independent genetic studies.

All identifiers, molecular links, ontology memberships and genetic coordinates
are fictional. Genetic summary statistics are NOT estimated from the cohort.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

VERSION = 'synthetic-v1'
TRAITS = {'SBP': 'mmHg', 'HbA1c': 'percentage points'}
DEFAULT_COVARIATES = ['age', 'sex', 'BMI', 'site']
LAYERS = ['expression', 'mirna', 'methylation']


def simulate(seed=2026, n=600):
    rng = np.random.default_rng(seed)
    ids = [f'SIM_PERSON_{i:04d}' for i in range(n)]
    age, sex, bmi = rng.normal(55, 9, n), rng.integers(0, 2, n), rng.normal(27, 4, n)
    site = rng.choice(['Site_A', 'Site_B', 'Site_C'], n)
    latent = rng.normal(size=(n, 4))
    p = rng.normal(size=(n, 60))
    p[:, :10] += .7 * latent[:, [0]]
    p[:, 10:20] += .8 * latent[:, [1]]
    p[:, 1] = rng.normal(size=n)  # independent negative-effect example
    p = (p - p.mean(axis=0)) / p.std(axis=0, ddof=1)
    sbp = 120 + .5*(age-55) + .7*(bmi-27) + 1.5*sex + (site == 'Site_C')*2
    sbp += p[:, :4] @ np.array([5., -4., 3., 2.]) + rng.normal(0, 7, n)
    hba = 5.5 + .006*(age-55) + .03*(bmi-27)
    hba += p[:, 10:14] @ np.array([.35, -.25, .2, .15]) + rng.normal(0, .35, n)
    phenotype = pd.DataFrame({'sample_id': ids, 'age': age, 'sex': sex,
        'BMI': bmi, 'site': site, 'SBP': sbp, 'HbA1c': hba}).set_index('sample_id')
    proteins = pd.DataFrame(p, index=phenotype.index,
                            columns=[f'SIM_P{i+1:03d}' for i in range(60)])
    expression = pd.DataFrame(.65*p + rng.normal(0, .7, p.shape), index=phenotype.index,
                              columns=[f'SIM_GENE_{i+1:03d}' for i in range(60)])
    mirna_values = rng.normal(size=(n, 24))
    for i in range(20):
        mirna_values[:, i] -= .6*p[:, i]
    mirna = pd.DataFrame(mirna_values, index=phenotype.index,
                         columns=[f'SIM_miR_{i+1:03d}' for i in range(24)])
    methylation = rng.normal(size=(n, 120))
    for i in range(60):
        methylation[:, 2*i] -= .7*expression.iloc[:, i].to_numpy()
    methylation = pd.DataFrame(expit(methylation), index=phenotype.index,
                               columns=[f'SIM_CpG_{i+1:03d}' for i in range(120)])
    links = []
    for i, protein in enumerate(proteins):
        gene = expression.columns[i]
        links.append(dict(protein=protein, feature=gene, layer='expression', gene=gene,
            link_type='synthetic encoding-gene mapping', source='SIMULATED annotation v1'))
        for j in [2*i, 2*i+1]:
            links.append(dict(protein=protein, feature=methylation.columns[j], layer='methylation',
                gene=gene, link_type='synthetic promoter annotation' if j%2 == 0 else 'synthetic gene-body annotation',
                source='SIMULATED annotation v1'))
        if i < 20:
            links.append(dict(protein=protein, feature=mirna.columns[i], layer='mirna', gene=gene,
                link_type='synthetic target hypothesis; NOT experimental validation',
                source='SIMULATED annotation v1'))
    # Deliberate failed and incomplete-case examples, after generating traits.
    proteins['SIM_P060'] = 1.
    proteins.iloc[:240, 58] = np.nan
    for frame in [proteins, expression, mirna, methylation]:
        values = frame.to_numpy(copy=True)
        values[rng.random(values.shape) < .015] = np.nan
        frame.iloc[:, :] = values
    gene_names = expression.columns.tolist()
    terms = {
        'SIM_GO:001': ('SIMULATED vascular signaling', gene_names[:10]),
        'SIM_GO:002': ('SIMULATED metabolic regulation', gene_names[10:20]),
        'SIM_GO:003': ('SIMULATED immune response', gene_names[20:35]),
        'SIM_GO:004': ('SIMULATED matrix organization', gene_names[30:45]),
        'SIM_GO:005': ('SIMULATED RNA processing', gene_names[40:]),
        'SIM_GO:006': ('SIMULATED cellular transport', gene_names[::3]),
        'SIM_GO:007': ('SIMULATED stress response', gene_names[::4]),
        'SIM_GO:008': ('SIMULATED signal integration', gene_names[2::5]),
    }
    annotations = pd.DataFrame([{'term': term, 'name': name, 'gene': gene,
        'source': 'FICTIONAL gene sets; NOT Gene Ontology database'}
        for term, (name, genes) in terms.items() for gene in genes])
    genetics = simulate_genetics(seed+1, proteins.columns.tolist())
    return dict(proteins=proteins, phenotype=phenotype, expression=expression, mirna=mirna,
        methylation=methylation, links=pd.DataFrame(links), ontology=annotations,
        genetics=genetics, metadata=dict(version=VERSION, seed=seed, n=n, synthetic=True,
        tissue='SIMULATED blood', ancestry='SIMULATED homogeneous population',
        genome_build='SIM_BUILD_1 (fictional)', independent_participants=True,
        genetic_note='Independent artificial summary-statistic studies, not derived from cohort data.'))


def simulate_genetics(seed, proteins):
    rng = np.random.default_rng(seed)
    regions = {}
    m, rho = 101, .85
    ld = rho ** np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    for i, protein in enumerate(proteins):
        scenario = i % 10
        if scenario == 3:  # absent pQTL resource
            continue
        positions = (i+1)*1_000_000 + np.arange(m)*1_000
        snps = [f'SIM_SNP_P{i+1:03d}_{j:03d}' for j in range(m)]
        def noise():
            z = rng.normal(size=m)
            for j in range(1, m):
                z[j] = rho*z[j-1] + np.sqrt(1-rho*rho)*z[j]
            return z
        sx = .02
        exposure_effect = .025 if scenario == 5 else .32
        bx = exposure_effect*ld[:, 40] + sx*noise()
        pqtl = pd.DataFrame(dict(snp=snps, position=positions, beta=bx, se=sx,
            effect_allele='A', other_allele='C', eaf=.3, N=12000, cis=True))
        if scenario == 6:
            pqtl['other_allele'] = 'T'  # palindromic QC branch
        pqtl['p_value'] = 2*norm.sf(np.abs(bx/sx))
        outcomes = {}
        for trait in TRAITS:
            sy = .08 if trait == 'SBP' else .008
            target = 76 if scenario == 2 else 40  # distinct/shared signal
            effect = (2.0 if trait == 'SBP' else .15)
            effect *= -1 if scenario == 1 else 1
            if scenario == 4:
                effect = 0  # pQTL-only region
            by = effect*ld[:, target] + sy*noise()
            frame = pd.DataFrame(dict(snp=snps, position=positions, beta=by, se=sy,
                effect_allele='A', other_allele='T' if scenario == 6 else 'C', eaf=.3, N=50000))
            if scenario == 1:
                frame['effect_allele'], frame['other_allele'] = 'C', 'A'
                frame['beta'], frame['eaf'] = -frame['beta'], .7
            frame['p_value'] = 2*norm.sf(np.abs(frame.beta/sy))
            if scenario != 7:
                outcomes[trait] = frame
        regions[protein] = dict(pqtl=pqtl, outcomes=outcomes, ld=ld.copy(), snp_order=snps,
            source='SIMULATED independent genetic studies', build='SIM_BUILD_1',
            scenario={0:'shared',1:'shared, reversed outcome alleles',2:'distinct signals',
                4:'pQTL only',5:'weak pQTL',6:'palindromic',7:'missing outcome'}.get(scenario,'shared'))
    return regions
