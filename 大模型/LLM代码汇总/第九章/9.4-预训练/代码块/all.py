# load necessary components
from trafilatura import fetch_url, extract
# download a web page
url = 'https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/'
downloaded = fetch_url(url)
downloaded is None  
# assuming the download was successfulFalse# extract information from HTML
result = extract(downloaded)
print(result)

import langid
text = "Tudo bem?"
language = langid.classify(text)[0]
print(language)

def compute_stats_single(self, sample):
    # check if it's computed already
    if StatsKeys.lang in sample[
            Fields.stats] and StatsKeys.lang_score in sample[Fields.stats]:
        return sample

    text = sample[self.text_key].lower().replace('\n', ' ')
    ft_model = get_model(self.model_key)
    if ft_model is None:
        err_msg = 'Model not loaded. Please retry later.'
        logger.error(err_msg)
        raise ValueError(err_msg)
    pred = ft_model.predict(text)
    lang_id = pred[0][0].replace('__label__', '')
    lang_score = pred[1][0]

    sample[Fields.stats][StatsKeys.lang] = lang_id
    sample[Fields.stats][StatsKeys.lang_score] = lang_score

    return sample

