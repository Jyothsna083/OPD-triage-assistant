import urllib.request
import re

html = urllib.request.urlopen("http://127.0.0.1:5000/").read().decode("utf-8", "ignore")
links = re.findall(r'<a href="[^"]*patient[^"]*"[^>]*>[\s\S]*?</a>', html)
print(f"Total links found: {len(links)}")
for l in links[:5]:
    print("Link:", l.strip())
