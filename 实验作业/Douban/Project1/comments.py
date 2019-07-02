import re
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib import request,error

#headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
movie_url = "https://movie.douban.com/cinema/nowplaying/beijing/"
#req = urllib.request.Request(url=movie_url, headers=headers)
    
def getbs(url):
    try:
        html=urlopen(url)
        bsobj=BeautifulSoup(html,'html.parser') #指定Beautiful的解析器为“html.parser”
        html.close()
    except (HTTPError,URLError):
        return None
    return bsobj

def getFilmList(url):
    bsobj = getbs(url)
    if bsobj == None: return
    lists = bsobj.find('ul', {'class': 'lists'}).findAll('li', {'class': 'list-item'})
    return lists

def getReview(url,n): #n:需要爬取的页数
    lists = getFilmList(url)
    print('电影总数：'+str(len(lists)))
    f = open('review.txt', 'w', encoding='utf-8')
    for i in range(30):
        print('开始爬取第%d个电影，电影名为《%s》'%(i+1,lists[i]['data-title']))
        reviewUrl='https://movie.douban.com/subject/'+lists[i]['id']+'/comments?status=P'
        text=[]
        for j in range(n):
            reviewBSobj = getbs(reviewUrl)
            if reviewBSobj == None:return
            comments=reviewBSobj.findAll('div',{'class':'comment-item'})
            print('-----第%d页-----' % (j+1))
            for comment in comments:
                star=comment.find('span',{'class':re.compile(r'^allstar(.*?)')})
                if star is None:continue
                print(star)
                if int(star['class'][0][7])>2:target=1
                else:target=0
                content=comment.p.get_text().strip()
                pattern = re.compile(r'[\u4e00-\u9fa5]+')
                filterdata = re.findall(pattern, content)
                cleaned_comments = ''.join(filterdata)
                text.append(str(target)+' '+cleaned_comments+'\n')
            print('-------------')
            reviewUrl='https://movie.douban.com/subject/'+lists[i]['id']+'/comments?start='+str(20*j+20)+'&limit=20&sort=new_score&status=P'
        f.writelines(text)
    f.close()

getReview(movie_url,10)
