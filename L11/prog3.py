# HTML structure Splitter
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<html>
<body>
  <h1>Main Title</h1>
  <p>Intro paragraph</p>
  <h2>Section 1</h2>
  <p>Content of section 1</p>
  <h2>Section 2</h2>
  <p>Content of section 2</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_string)

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)