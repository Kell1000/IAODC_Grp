import os
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

nb_pages = 1000

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


def download_file(pdf_url, filename):
    response = session.get(pdf_url, timeout=30)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)


for n_page in range(1, nb_pages + 1):

    # Replace this with the actual CORE search URL.
    url = f"https://core.ac.uk/search?q=YOUR_QUERY&page={n_page}"

    print(f"HTTP GET: {url}")

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find links containing PDFs
        all_urls = soup.find_all("a", href=True)

        for link in all_urls:

            href = link["href"]

            if ".pdf" not in href.lower():
                continue

            # Convert relative URL to absolute URL
            pdf_url = urllib.parse.urljoin(url, href)

            print(f"Downloading: {pdf_url}")

            try:
                pdf_response = session.get(
                    pdf_url,
                    timeout=60,
                    allow_redirects=True
                )
                pdf_response.raise_for_status()

                # Get filename from final URL
                parsed_url = urllib.parse.urlparse(pdf_response.url)
                filename = os.path.basename(parsed_url.path)

                # If the URL doesn't contain a filename
                if not filename or not filename.lower().endswith(".pdf"):
                    filename = f"paper_{n_page}.pdf"

                # Remove unsafe characters
                filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

                download_file(pdf_response.url, filename)

                print(f"Saved: {filename}")

            except requests.RequestException as e:
                print(f"PDF download failed: {e}")

    except requests.RequestException as e:
        print(f"Page request failed: {e}")
