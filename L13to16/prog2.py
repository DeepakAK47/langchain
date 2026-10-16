# use of DuckDuckGoSearch 
from langchain_community.tools import DuckDuckGoSearchResults
search = DuckDuckGoSearchResults()

result = search.invoke("Who is Virat Kohli?")
print(result)