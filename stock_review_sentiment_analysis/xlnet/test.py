
review_count= '30'
if '.'not in review_count:
    shu = float(review_count)
else:
    index = review_count.index("万")
    shu = float(review_count[:index])*10000
print(shu)

