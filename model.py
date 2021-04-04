import pandas as pd


# Class to retrieve JD
# Attributes: Company, location, profession, designation, link
class JobsDesc:
    def __init__(self):
        self.data = pd.read_csv(r"data/jd_combined.csv", sep=";")


# Class to retrieve reviews
# Attributes: Company, rating_overall, recommends, pros and cons
class Reviews:
    def __init__(self):
        self.data = pd.read_csv(r"data/review_combined.csv", sep=";", header=0)

    def get_company_overall_rating(self, company):
        company_pd = self.data[self.data["company"] == company]
        rating = sum(list(company_pd["rating_overall"]))/len(company_pd["rating_overall"])
        return rating


class Sentiments:
    def __init__(self):
        self.data = pd.read_csv(r"data/sentiment.csv", sep=",", header=0)

    def get_company_sentiment(self, company):
        return self.data[self.data["company"] == company]


if __name__ == '__main__':
    jd = JobsDesc()
    print(jd.data)

    r = Reviews()
    print(r.data)

