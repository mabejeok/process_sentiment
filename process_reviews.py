import string
import numpy as np
import pandas as pd
import nltk
import joblib
import fasttext
from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob
from model import Reviews
from datetime import datetime
from nltk.corpus import stopwords, wordnet
from nltk import WordNetLemmatizer, pos_tag, word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


# Data cleaning
def remove_stopwords(words, blacklist=None, whitelist=None):
    """

    :param words: Word
    :param blacklist:
    :param whitelist:
    :return:
    """
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []

    # Sample whitelist, we will keep these words so that will not affect our sentiment analysis
    # whitelist = ["n't", "not", "no", "never", "nothing", "nowhere", "no one", "no one", "none", "haven't",
    #              "hasn't", "hasnt", "hadn't", "hadnt", "can't", "cant", "couldn't", "couldnt", "shouldn't",
    #              "shouldnt","won't","wont","wouldn't","wouldnt","don't","dont","doesn't","doesnt","didn't",
    #              "didnt", "isnt", "isn't", "aren't", "arent", "aint", "ain't", "hardly", "seldom"]

    stopwords_list = stopwords.words('english') + blacklist

    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return clean_words


def wnl_lemma(tokens):
    wnl = WordNetLemmatizer()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    toks = []
    toks_tag = pos_tag(tokens)
    for item in toks_tag:
        word = item[0]
        tag = item[1][0].upper() # first letter of pos_tag generated
        tag = tag_dict.get(tag, wordnet.NOUN)
        toks.append(wnl.lemmatize(word, tag))
    return toks


def clean_title_details(input_text):
    itant_index = input_text.find("ltant")
    pros_index = input_text.find("Pros\n")
    if "helpfu" in input_text[:-10]:
        input_text = input_text[:-input_text.rfind("helpfu")]
    if itant_index != -1 and pros_index != -1:
        before_itant = input_text[: itant_index]
        after_pros = input_text[pros_index:]
        cleaned_string = before_itant + " " + after_pros
        cleaned_string = cleaned_string.replace("Pros\n", " ")
        cleaned_string = cleaned_string.replace("Cons\n", " ")
        return cleaned_string
    else:
        return input_text


def preprocess(input_text, blacklist=None, whitelist=None):
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    input_text = input_text.replace("\n", " ")
    toks = word_tokenize(input_text)
    toks = [t for t in toks if not t.isnumeric()] # remove tokens that are numbers
    toks = [t.lower() for t in toks if t not in string.punctuation + "’“”" + "''"]
    toks = wnl_lemma(toks)
    toks = remove_stopwords(toks, blacklist, whitelist)

    return toks


def get_sentiment(text, analyser):
    if analyser == "textblob":
        score_min = -1
        score_max = 1
        res = TextBlob(text).sentiment.polarity
        senti = np.round(res, 2)
    elif analyser == "fasttext":
        score_min = 1
        score_max = 5
        model = fasttext.load_model("sentiment_analysers/sst5.bin")
        labels, probabilities = model.predict(text.replace("\n", " "), 1)
        senti = int(labels[0][-1])
    elif analyser == "flair":
        score_min = 1
        score_max = 5
        model = TextClassifier.load("sentiment_analysers/best-model.pt")
        doc = Sentence(text)
        model.predict(doc)
        senti = int(doc.labels[0].value)
    else:
        score_min = -1
        score_max = 1
        analyzer = SentimentIntensityAnalyzer()
        res = analyzer.polarity_scores(text)
        senti = np.round(res['compound'], 2)
    senti = (senti - score_min) / (score_max - score_min)
    return senti


def get_cat(curr_aspect_dict, sentence):
    category=[]
    for aspect, word_list in curr_aspect_dict.items():
        for word in word_list:
            if word in sentence:
                category.append(aspect)
    if len(category) == 0 : category.append("other")
    return list(set(category))


def get_sent_cat(curr_aspect_dict, sent_list, analyser="vader"):
    review_dict = {"other": []}
    score_dict = {}
    for key in curr_aspect_dict.keys():
        review_dict[key] = []

    for sent in sent_list:
        all_cats = get_cat(curr_aspect_dict, sent)
        senti = get_sentiment(sent, analyser)
        for cat in all_cats:
            review_dict[cat].append(senti)

    for aspect, value_list in review_dict.items():
        if value_list:
            score_dict[aspect] = sum(value_list) / len(value_list)
        else:
            score_dict[aspect] = 0

    return score_dict


def normalize(num: float, max: float, min:float):
    average = (min + max)/2
    range = (max - min)/2
    normalized_num = (num - average)/range
    return round(normalized_num, 2)


def output_sentiment_file(selected_csv_path: str):
    final_pd_flair = pd.read_csv(selected_csv_path)
    # Here we are scaling the score to be in % so that it is easily displayed for users.
    # We will store the csv so that the app can grab the scores easily instead of recomputing.
    sent_pd = final_pd_flair[
        ["company", "other", "management", "culture", "pay", "growth", "environment", "balance"]].copy()
    new_cols = ["other", "management", "culture", "pay", "growth", "environment", "balance"]
    for col in new_cols:
        sent_pd[col] = sent_pd[col].map(lambda x: x * 100)

    sent_pd.to_csv("sentiment.csv", index=False)


if __name__ == '__main__':
    start_time = datetime.now()

    # Display full pandas table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    aspect_dict = {
        "management": ["management", "administration", "steering", "authority",
                       "oversight", "mismanagement", "supervis", "manager",
                       "director", "senior", "review", "appraisal", "assess",
                       "help", "clear", "communication",
                       "supportive", "ceo", "meeting", "boss", "reassur", "approv",
                       "recognize", "recognise", "board", "superior", "headquarter",
                       "mentor"],
        "culture": ["culture", "team", "staff", "colleague", "friendly", "senior",
                    "fast", "slow", "politic", "fun", "competitive",
                    "young", "pace", "smart", "talent", "organization",
                    "organisation", "communication", "clear", "supportive", "old",
                    "meeting", "fresh", "innovat", "morale", "conference",
                    "office", "crew", "toxic", "speed", "prompt", "bureaucracy",
                    "civil service", "public service", "personnel", "department",
                    "profit", "autonomy", "melting"],
        "pay": ["pay", "salary", "benefit", "bonus", "progression", "promotion",
                "compensation", "perk", "reward", "package", "wage", "bonus",
                "insurance", "bucks", "budget", "cash"],
        "growth": ["opportunity", "learn", "career", "growth", "grow", "training",
                   "progression", "promotion", "skill", "exposure", "performance",
                   "talent", "promote", "stable", "money", "profession", "achievement",
                   "routine"],
        "balance": ["balance", "time", "hour", "life", "flexible", "free",
                    "family", "load", "flexibility", "home", "stress",
                    "stable", "pressure", "meeting", "shift", "conference", "leave"],
        "environment": ["environment", "place", "office", "tech", "system",
                        "fun", "location", "food", "resource", "local",
                        "global", "atmosphere", "innovation", "work", "surround",
                        "traffic", "intelligen", "automat", "town", "home", "convenien",
                        "comfort"]
    }

    # Lemmatize words in the aspect_dict
    for aspect, word_list in aspect_dict.items():
        toks_tag = pos_tag(word_list)
        word_list = wnl_lemma(word_list)

    r = Reviews().data.head(5)

    r[["review_title", "pros", "cons"]] = r[["review_title", "pros", "cons"]].fillna("")
    r["title_details"] = r["review_title"] + " " + r["pros"] + " " + r["cons"]

    r_sentiment = r[["company", "title_details"]].copy()

    # For each title_details, clean up those with "ltant" and make sentences lower case. Then split into list of
    # sentences (column "td_sents"). For each sentence, we find out the sentiment + the categories.
    # After which, we will add up all the scores per sentence per category to have a dictionary of
    # {"cat1": 0.8, "cat2": 0.9 etc}

    r_sentiment["title_details"] = r_sentiment["title_details"].map(lambda x: clean_title_details(x).lower())
    r_sentiment["td_sents"] = r_sentiment["title_details"].map(lambda x: sent_tokenize(x))
    r_sentiment["td_sent_cat_vader"] = r_sentiment["td_sents"].map(lambda x: get_sent_cat(aspect_dict, x))
    r_sentiment["td_sent_cat_textblob"] = r_sentiment["td_sents"].map(lambda x: get_sent_cat(aspect_dict, x,
                                                                                             analyser="textblob"))
    r_sentiment["td_sent_cat_fasttext"] = r_sentiment["td_sents"].map(lambda x: get_sent_cat(aspect_dict, x,
                                                                                             analyser="fasttext"))
    r_sentiment["td_sent_cat_flair"] = r_sentiment["td_sents"].map(lambda x: get_sent_cat(aspect_dict, x, analyser="flair"))

    r_sentiment = r_sentiment.merge(r,
                                    on=["title_details", "company"])
    r_sentiment = r_sentiment[["company", "title_details", "td_sent_cat_vader", "td_sent_cat_textblob",
                               "td_sent_cat_fasttext", "td_sent_cat_flair", "rating_balance", "rating_culture",
                               "rating_career", "rating_comp", "rating_mgmt", "rating_overall"]]
    for col in ["rating_balance", "rating_culture", "rating_career", "rating_comp", "rating_mgmt", "rating_overall"]:
        r_sentiment[col] = r_sentiment[col].map(lambda x: (x - r_sentiment[col].min()) / (r_sentiment[col].max() - r_sentiment[col].min()))

    # Store the reviews and their scores to analyze later.
    r_sentiment.to_csv("sent_senti_rating.csv", index=False)

    # Here we are trying to add up aspect scores for the companies given their reviews.
    company_sent_pd = pd.DataFrame(list(r_sentiment["company"].unique()), columns=['company'])
    company_sent_pd["other"] = 0
    for aspect in aspect_dict.keys():
        company_sent_pd[aspect] = 0

    for classi_type in ["vader", "textblob", "fasttext", "flair"]:

        for company in r_sentiment["company"].unique():
            company_pd = r_sentiment[r_sentiment["company"] == company]
            overall_review_dict = {"other": []}
            overall_score_dict = {}
            for aspect in aspect_dict.keys():
                overall_review_dict[aspect] = []
            for row in company_pd.itertuples(index=False):
                sent_dict = getattr(row, f"td_sent_cat_{classi_type}")
                for aspect, val in sent_dict.items():
                    overall_review_dict[aspect].append(val)
            for aspect, value_list in overall_review_dict.items():
                if value_list:
                    company_sent_pd.loc[company_sent_pd["company"] == company, aspect] = round(sum(value_list) / len(value_list), 2)
                else:
                    company_sent_pd.loc[company_sent_pd["company"] == company, aspect] = 0

        # We will then merge the company's aspect scores derived from sentiment analysis vs those
        # that are organically rated by the employees on the websites to compare.
        r_rating = (r[["company", "rating_balance", "rating_culture", "rating_career", "rating_comp", "rating_mgmt",
                       "rating_overall"]]
                    .dropna(how="all"))
        r_rating = (r_rating
                    .groupby("company")["rating_balance", "rating_culture", "rating_career", "rating_comp", "rating_mgmt",
                                        "rating_overall"]
                    .apply(lambda x: x.mean()))
        final_pd = pd.merge(r_rating, company_sent_pd, on="company")
        final_pd = final_pd.drop_duplicates()

        # Here we scale all the ratings to be within [0, 1]. This is because ratings on website is between 0 to 5 while
        # sentiment scores are between -1 to 1.
        all_cols = list(final_pd.columns)
        all_cols.remove("company")
        for col in all_cols:
            final_pd[col] = final_pd[col].map(lambda x: (x - final_pd[col].min()) / (final_pd[col].max() - final_pd[col].min()))

        # Now, we calculate the difference between the rating from website and our sentiment score
        # We will keep this for future analysis.
        final_pd["diff_balance"] = (final_pd[["rating_balance", "balance"]]
                                    .apply(lambda x: 100 * (x["balance"] - x["rating_balance"])/x["rating_balance"], axis=1))
        final_pd["diff_culture"] = (final_pd[["rating_culture", "culture"]]
                                    .apply(lambda x: 100 * (x["culture"] - x["rating_culture"]) / x["rating_culture"], axis=1))
        final_pd["diff_pay"] = (final_pd[["rating_comp", "pay"]]
                                    .apply(lambda x: 100 * (x["pay"] - x["rating_comp"]) / x["rating_comp"], axis=1))
        final_pd["diff_career"] = (final_pd[["rating_career", "growth"]]
                                .apply(lambda x: 100 * (x["growth"] - x["rating_career"]) / x["rating_career"], axis=1))
        final_pd["diff_mgmt"] = (final_pd[["rating_mgmt", "management"]]
                                     .apply(lambda x: 100 * (x["management"] - x["rating_mgmt"]) / x["rating_mgmt"], axis=1))
        final_pd.to_csv(f"final_pd_{classi_type}.csv", index=False)

    # From LIME's visualisation, selected Vader as the final analyser so we will use this to generate sentiment.csv
    output_sentiment_file("data/final_pd_vader.csv")

    print(datetime.now() - start_time)


