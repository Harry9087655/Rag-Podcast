import sys
from pprint import pprint, pformat
import json
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import feedparser
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    filename="explore.log", filemode="w")
url = "https://lexfridman.com/feed/podcast/"

feed = feedparser.parse(url)
f = feed.feed
logger.info(f"Number of entries: {len(feed.entries)}")
logger.info(f"Available fields in the first entry: {pformat(feed.entries[0].keys())}")
logger.info(
    pformat(
        feed.entries[0],
        indent=2,
        width=100,
        depth=3,
    )
)

print(feed.entries[0].get("enclosures", []))