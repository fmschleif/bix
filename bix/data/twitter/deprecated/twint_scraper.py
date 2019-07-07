import twint
from twint.output import tweets_object

hashtag = 'brexit' # 'peoplesvote'
num = 10


c = twint.Config()

c.Search = '#' + hashtag
c.Lang = 'en'
#c.Username = "noneprivacy"
#c.Custom["tweet"] = ["id"]
#c.Custom["user"] = ["bio"]
#c.Custom['id'] = ['id']
c.Custom['tweet'] = ['tweet']
c.Limit = num
c.Store_csv = True
c.Output = 'hashtag_' + hashtag


c.Store_object = True
twint.output.tweets_object = []
twint.run.Search(c)
#print(f"test {tweets_object[0].tweet}")
