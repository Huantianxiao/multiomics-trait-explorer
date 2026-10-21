"""Deterministic local calculations. Synthetic studies only; no LLM statistics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp, softmax
from scipy.stats import norm, chi2, hypergeom
from statsmodels.stats.multitest import multipletests

ASSOCIATION_COLUMNS = ['feature', 'n', 'beta', 'se', 'ci_low', 'ci_high', 'p_value', 'status', 'q_value', 'significant']


def correct_fdr(table, fdr=.05):
    result = table.copy()
    result['q_value'] = np.nan
    tested = result.p_value.notna() & np.isfinite(result.p_value)
    if tested.any():
        result.loc[tested, 'q_value'] = multipletests(result.loc[tested, 'p_value'], method='fdr_bh')[1]
    result['significant'] = result.q_value.lt(fdr)
    return result


def covariate_design(phenotype, covariates):
    if phenotype.index.duplicated().any():
        raise ValueError('Participant IDs must be unique.')
    cov = phenotype.loc[:, covariates].copy()
    categorical = [c for c in cov if c == 'site']
    if categorical:
        cov = pd.get_dummies(cov, columns=categorical, drop_first=True, dtype=float)
    return cov.astype(float)


def fit_feature(feature, outcome, cov, min_n=50, max_missing=.2):
    frame = pd.concat([feature.rename('_feature'), outcome.rename('_outcome'), cov], axis=1).replace([np.inf, -np.inf], np.nan)
    frame = frame.reindex(outcome.index)
    row = dict(n=0, beta=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan, status='')
    if frame['_feature'].isna().mean() > max_missing:
        row['status'] = 'skipped: excessive feature missingness'
        return row
    clean = frame.dropna().copy()
    row['n'] = len(clean)
    if len(clean) < min_n:
        row['status'] = 'skipped: insufficient complete samples'
        return row
    sd = clean['_feature'].std(ddof=1)
    if sd < 1e-12:
        row['status'] = 'skipped: constant feature'
        return row
    clean['_feature'] = (clean['_feature']-clean['_feature'].mean())/sd
    x = sm.add_constant(clean.drop(columns='_outcome'), has_constant='add')
    if len(clean) <= x.shape[1]+2 or np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        row['status'] = 'skipped: rank-deficient design'
        return row
    model = sm.OLS(clean['_outcome'], x).fit(cov_type='HC3', use_t=True)
    lo, hi = model.conf_int().loc['_feature']
    row.update(beta=float(model.params['_feature']), se=float(model.bse['_feature']),
        ci_low=float(lo), ci_high=float(hi), p_value=float(model.pvalues['_feature']), status='tested')
    return row


def scan(frame, phenotype, trait, covariates, fdr=.05):
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError('Molecular sample IDs and feature names must be unique.')
    cov = covariate_design(phenotype, covariates)
    rows = [dict(feature=feature, **fit_feature(frame[feature], phenotype[trait], cov)) for feature in frame]
    return correct_fdr(pd.DataFrame(rows), fdr).sort_values('p_value', na_position='last').reset_index(drop=True)


def molecular_evidence(data, layer, candidates, plan):
    """Test all layer features and all registered links BEFORE candidate filtering.

    This keeps BH families explicit and avoids presenting only a post-screen
    handful of tests as a complete layer-wide discovery analysis.
    """
    frame = data[layer]
    if layer == 'methylation':
        frame = np.log2(frame.clip(1e-6, 1-1e-6)/(1-frame.clip(1e-6, 1-1e-6)))
    results = scan(frame, data['phenotype'], plan['trait'], plan['covariates'], plan['fdr'])
    links = data['links'].loc[data['links'].layer.eq(layer)].copy()
    cov = covariate_design(data['phenotype'], plan['covariates'])
    pair_rows = []
    for link in links.itertuples():
        # Protein is standardized so the partial association has SD/SD units.
        protein = data['proteins'][link.protein]
        protein = (protein-protein.mean())/protein.std(ddof=1) if protein.std(ddof=1) > 1e-12 else protein*0
        pair_rows.append(dict(protein=link.protein, feature=link.feature,
            **fit_feature(frame[link.feature], protein, cov)))
    pairs = correct_fdr(pd.DataFrame(pair_rows), plan['fdr'])
    evidence = links.merge(pairs, on=['protein','feature'], validate='one_to_one')
    evidence = evidence.rename(columns={c: f'protein_link_{c}' for c in ['n','beta','se','ci_low','ci_high','p_value','q_value','significant','status']})
    evidence = evidence.merge(results.rename(columns={c:f'trait_{c}' for c in results if c != 'feature'}), on='feature', validate='many_to_one')
    evidence = evidence.loc[evidence.protein.isin(candidates)].reset_index(drop=True)
    return results, evidence


def harmonize_region(region, trait):
    """Dense common-SNP table, before any instrument selection or LD pruning."""
    x = region['pqtl']
    y = region['outcomes'].get(trait)
    if y is None:
        return pd.DataFrame()
    if x.snp.duplicated().any() or y.snp.duplicated().any():
        raise ValueError('Duplicate SNP statistics in a region.')
    merged = x.merge(y, on=['snp','position'], suffixes=('_x','_y'), validate='one_to_one')
    rows = []
    for r in merged.itertuples():
        pair = (r.effect_allele_x, r.other_allele_x)
        other = (r.effect_allele_y, r.other_allele_y)
        sign = 1 if pair == other else -1 if pair == other[::-1] else None
        status = 'aligned'
        if set(pair) in [{'A','T'}, {'C','G'}]:
            status = 'excluded: palindromic'
        elif sign is None:
            status = 'excluded: incompatible alleles'
        elif not np.isfinite([r.beta_x,r.beta_y,r.se_x,r.se_y]).all() or min(r.se_x,r.se_y) <= 0:
            status = 'excluded: invalid summary statistics'
        rows.append(dict(snp=r.snp, position=r.position, beta_x=r.beta_x, se_x=r.se_x,
            beta_y=r.beta_y*(sign or 1), se_y=r.se_y, p_x=r.p_value_x, p_y=r.p_value_y,
            cis=r.cis, F=(r.beta_x/r.se_x)**2 if r.se_x > 0 else np.nan,
            alignment='flipped' if sign == -1 else 'same', status=status))
    return pd.DataFrame(rows)


def select_instruments(dense, region, r2=.001):
    qc = dense.copy()
    qc['decision'] = qc.status
    passed = qc.status.eq('aligned')
    qc.loc[passed, 'decision'] = 'excluded: weak or nonsignificant pQTL'
    selected = []
    eligible = qc.loc[passed & qc.cis & qc.p_x.le(5e-8) & qc.F.ge(10)].sort_values('p_x')
    order = {s:i for i,s in enumerate(region['snp_order'])}
    for idx, row in eligible.iterrows():
        i = order[row.snp]
        if any(region['ld'][i, order[qc.loc[j,'snp']]]**2 > r2 for j in selected):
            qc.loc[idx, 'decision'] = 'excluded: LD with selected simulated instrument'
        else:
            selected.append(idx)
            qc.loc[idx, 'decision'] = 'eligible'
    return qc


def mr_estimate(instruments):
    bx, sx, by, sy = [instruments[c].to_numpy(float) for c in ['beta_x','se_x','beta_y','se_y']]
    n = len(bx)
    if not n:
        raise ValueError('No eligible instruments.')
    if n == 1:
        beta = by[0]/bx[0]
        se = np.sqrt(sy[0]**2/bx[0]**2 + by[0]**2*sx[0]**2/bx[0]**4)
        method, q, qp = 'Wald ratio; independent-sample delta SE', np.nan, np.nan
    else:
        w = 1/sy**2
        denominator = np.sum(w*bx**2)
        beta = np.sum(w*bx*by)/denominator
        se = np.sqrt(1/denominator)
        q = float(np.sum(w*(by-beta*bx)**2))
        qp = float(chi2.sf(q,n-1))
        method = 'Fixed-effect IVW; NOME'
    return dict(n_snps=n, method=method, beta=float(beta), se=float(se),
        ci_low=float(beta-1.95996398454*se), ci_high=float(beta+1.95996398454*se),
        p_value=float(2*norm.sf(abs(beta/se))), Q=q, Q_p_value=qp, status='estimated')


def coloc_abf(beta_x, se_x, beta_y, se_y, sd_y, p1=1e-4, p2=1e-4, p12=1e-5):
    """Wakefield ABFs and H0-H4 enumeration; one causal variant/trait/region.

    Quantitative effects: protein SD units, outcome in trait units. Effect-size
    prior SDs are 0.15*phenotype SD. H3 explicitly excludes shared-SNP pairs.
    This is a small Python implementation, not a call to the R coloc package.
    """
    bx,sx,by,sy = [np.asarray(v,float) for v in [beta_x,se_x,beta_y,se_y]]
    if len(bx) < 20 or len({len(v) for v in [bx,sx,by,sy]}) != 1:
        raise ValueError('Colocalization requires matching dense regional statistics (at least 20 SNPs in this demo).')
    if not np.isfinite(np.concatenate([bx,sx,by,sy])).all() or (sx<=0).any() or (sy<=0).any() or not np.isfinite(sd_y) or sd_y <= 0:
        raise ValueError('Invalid ABF inputs.')
    if not 0 < p12 <= min(p1,p2) < 1:
        raise ValueError('Invalid SNP priors.')
    def labf(b,s,w):
        v=s*s
        r=w/(v+w)
        return .5*(np.log1p(-r)+r*(b/s)**2)
    lx,ly = labf(bx,sx,.15**2), labf(by,sy,(.15*sd_y)**2)
    pairs = lx[:,None]+ly[None,:]
    np.fill_diagonal(pairs,-np.inf)
    log_h = np.array([0., np.log(p1)+logsumexp(lx), np.log(p2)+logsumexp(ly),
        np.log(p1)+np.log(p2)+logsumexp(pairs), np.log(p12)+logsumexp(lx+ly)])
    pp=softmax(log_h)
    return {f'PP_H{i}':float(pp[i]) for i in range(5)}, softmax(lx+ly)


def go_enrichment(hits, background, annotations, fdr=.05):
    """Over-representation with fictional gene sets; all terms in BH family."""
    background, hits = set(background), set(hits)&set(background)
    if not hits:
        return pd.DataFrame(columns=['term','name','overlap','hit_count','term_count','background_count','p_value','q_value','significant','genes'])
    rows=[]
    for (term,name), part in annotations.groupby(['term','name']):
        genes=set(part.gene)&background
        overlap=hits&genes
        if not genes:
            continue
        rows.append(dict(term=term,name=name,overlap=len(overlap),hit_count=len(hits),
            term_count=len(genes),background_count=len(background),
            p_value=float(hypergeom.sf(len(overlap)-1,len(background),len(genes),len(hits))),
            genes=', '.join(sorted(overlap)), source='FICTIONAL GO-like gene sets; not real GO annotations'))
    return correct_fdr(pd.DataFrame(rows),fdr).sort_values('p_value').reset_index(drop=True)
