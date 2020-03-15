import requests
url = 'http://0.0.0.0:5000/predict' 

#['TargetAge', 'HousingIndex', 'ppiscore', 'peopleAgesBelow17', 'MainIncomeSourceRegular', 'MemberLiterate', 'LoanCycle', 'CBR_CUR_BAL_HM', 'LoanAmountApproved', 'TotalMonthlyExp']

r = requests.post(url,json=[52,6,20,2,6,0,50000,4500])
print(r.json())

j = requests.post(url,json=[27,3,24,5,1,0,30000,8000])
print(j.json())
