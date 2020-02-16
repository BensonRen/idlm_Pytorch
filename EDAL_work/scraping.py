import scholarly
import pandas as pd
from tqdm import tqdm


energy_terms = [
    #'Transmission line',
    #'Electricity line',
    'Power line',
    'Energy infrastructure',
    'Electric infrastructure',
    #'Power',
    #'Energy'
    #'Generator',
    #'Coal',
    #'Oil',
    #'Natural Gas',
    # 'Geothermal',
    # 'Hydropower'
]

ml_terms = [
    'machine learning',
    'deep learning',
    'support vector machine',
    'random forest',
    'regression tree',
    'neural network'
]

rs_terms = [
    'remote sensing',
    'satellite',
    'aerial',
    'UAV',
    'unmanned aerial vehicle',
    'hyperspectral'
]


def quote(s):
    single_quote = '\''
    return single_quote + s + single_quote


for e in tqdm(energy_terms):
    print()
    results = []
    for m in ml_terms:
        for r in rs_terms:
            kw = ';'.join([quote(e), quote(m), quote(r)])
            search_query = scholarly.search_pubs_query(kw)
            i = 0
            while i < 20:
                try:
                    res = next(search_query)
                    i += 1
                    if hasattr(res, 'citedby'):
                        res.bib['citedby'] = res.citedby
                    else:
                        res.bib['citedby'] = 'NA'
                    res.bib['kw1'] = e
                    res.bib['kw2'] = m
                    res.bib['kw3'] = r
                    results.append(res.bib)
                except StopIteration:
                    break
    results_pd = pd.DataFrame.from_dict(results)
    results_pd.to_csv(f'./{e}.csv', index=False)
