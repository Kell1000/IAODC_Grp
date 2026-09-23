import urllib.request 
import requests
from bs4 import BeautifulSoup

nb_pages = 1000

def downoald_file(downoald_url, filename):
    response = urllib.request.urlopen(downoald_url)
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()

for n_page in range(1, nb_pages):
    url_i = 'https://core.ac.uk/search......'
    url = url_i = str(n_page)
    print('HTTP GET: %s', url)
    response = requests.get(url)
    # parse content
    content = BeautifulSoup(response.text, 'lxml')
    # extract URLs referencing PDF documents
    all_urls = content.find_all('a', href=True)#('figure)
    # loop over al URLs
    for url in all_urls:
        try:
            if 'pdf' in url['href']:
                #init PDF url
                pdf_url = ''
                #append base URL if no 'https' available in URL
                if 'https' not in url['href']:
                    pdf_url = 'https://core.ac.uk/' + url['href']
                else:
                    pdf_url = url['href']
                # make HTTP GET request to fetch PDF bytes
                    print('HTTP GET:%s', pdf_url)
                    url_patch = pdf_url
                    pdf_response = requests.get(pdf_url)
                # extract PDF file name
                    filename = urllib.request.unquote(pdf_response.url)
                    downoald_file(url_path, filename)
        