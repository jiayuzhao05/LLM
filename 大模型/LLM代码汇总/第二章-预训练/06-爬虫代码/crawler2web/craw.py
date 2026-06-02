# -*- coding: utf-8 -*-
import io
import requests
from bs4 import BeautifulSoup
import urllib
import ssl
import pandas as pd
import tqdm
ssl._create_default_https_context = ssl._create_unverified_context

def download_pdf(save_path, pdf_name, pdf_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/88.0.4324.146 Safari/537.36'
    }
    response = requests.get(pdf_url, headers=headers)
    bytes_io = io.BytesIO(response.content)
    with open("./事故报告_pdf/" + "%s.pdf" % pdf_name, mode='wb') as f:
        f.write(bytes_io.getvalue())
        print('%s.pdf,下载成功！' % (pdf_name))

def request_douban(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/88.0.4324.146 Safari/537.36'
    }

    try:
        response = requests.get(url=url, headers=headers, allow_redirects=False)
        response.encoding = response.apparent_encoding
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None

def askURL(url):
    head = { 
   "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36"}
    request=urllib.request.Request(url,headers=head)
    html=""
    try:
        response=urllib.request.urlopen(request)
        html=response.read().decode("utf-8")
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)

    return html

if __name__ == '__main__':
    save_path = './事故报告'
    urls = []
    for i in range(1, 117):
        urls.append(f"https://openstd.samr.gov.cn/bzgk/gb/std_list_type?r=0.03316401348913578&page={i}&pageSize=10&p.p1=1&p.p6=13&p.p90=circulation_date&p.p91=desc")

    names = []
    nums = []
    dodates = []
    # q = tqdm(urls, desc='Processing')
    for idx, url in enumerate(urls):
        html = askURL(url)
        bs = BeautifulSoup(html, 'html.parser')
        contents = bs.find('table', class_='table result_list table-striped table-hover').find_all('tr')
        for content in contents:
            try:
                th = content.find_all('td')
                nums.append(th[1].text.strip())
                names.append(th[3].text.strip())
                dodates.append(th[6].text.strip()[:10])
            except:
                pass
        print(f"完成第{idx}篇")

    df = pd.DataFrame(columns=["编号","名称", "实施日期"])
    for i in range(len(nums)):
        df.loc[i] = [nums[i], names[i], dodates[i]]
    
    df.to_csv('result.csv', index=False)