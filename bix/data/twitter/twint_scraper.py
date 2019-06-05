import twint

hashtag = 'brexit' # 'peoplesvote'
num = 10000


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

twint.run.Search(c)