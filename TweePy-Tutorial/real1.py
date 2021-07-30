#Twitter data full archive try-outs.

import tweepy

"""
API.search_full_archive(label, query, *, tag, fromDate, toDate, maxResults, next)
label = label of query -> tweetMining
query = for premium 1024, for standard 256. This parameter should include all portions of the rule/filter, including all operators, and portions of the
    rule should not be separated into other parameters of the query.

tag = 
fromDate=
toDate=
maxResults = Between 10 and 100 for sandbox or 500 for premium.
next = Next page.

This function returns 
"""

#Authentication 
api_key = "MZj6rr4yJh5veKzNy5XeHeQ5w"
api_key_secret= "Hcx4y46f9PTImLxzQwSNhjAlaE8Y1FriEg0AX7Z9Kb2FCJtaoh"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAPNNMAEAAAAAyK6dk4YQlYhui%2F0rE00pGEqapfM%3Dn2xsgM6qPBzSIbg1B4lcm6NEcsKLeVpJqDmBKAbIbOb2MvlSbD"
access_token = "760718278130008064-rUDSygDr1ZhGpykGt1Xr4ojgeVCSNwk"
access_token_secret = "yM3tEWetgr5fjc0tgRZzqJc2XpX62dexRDxF2rGMZnOAc"
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#Check connection
check = api.me()
print("This is for connection check: ", me.screen_name)

environment_name = "tweetMining"

#One request 
twits1 = api.search_full_archive()


