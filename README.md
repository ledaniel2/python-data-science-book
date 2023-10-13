# Learn Python for Data Science Book

This is the online companion repository to the book
**Learn Python for Data Science** (ASIN: B0C37SVQKZ) available from
[Amazon.com](https://www.amazon.com/dp/B0C37SVQKZ).

The majority of code samples in directory "samples"
should work without modification and are machine-extracted from the book;
some from later chapters need valid web API user credentials in order to be
run. The full list of libraries needed to run all of the code samples is
located in chapter 3.

Markdown sources for all of the chapters in the book are now available
to browse online. The command used to build the e-book under Windows 10 using
the "pandoc" program was:

```
..\pandoc-3.1.2\pandoc --epub-cover-image=cover2small.png --epub-title-page=false --highlight-style=code.theme --syntax-definition=plaintext.xml --metadata title="Learn Python for Data Science" --data-dir=..\pandoc-3.1.2\data -f gfm -t epub3 -o book-test.epub 00-overview.md 01-introduction.md 02-getting-started.md 03-libraries-data-science.md 04-working-data.md 05-data-cleaning.md 06-data-analysis.md 07-features.md 08-data-manipulation.md 09-data-visualization.md 10-model-evaluation.md 11-machine-learning.md 12-advanced-techniques.md 13-deep-learning.md 14-time-series-analysis.md 15-natural-language.md 16-web-scraping.md 17-working-apis.md 18-ethics.md 19-best-practices.md 20-conclusion.md
```

Note that not all of the resources referenced above are currently available online.

## Book Contents

0. [Overview](https://github.com/ledaniel2/python-data-science-book/blob/main/00-overview.md)
1. [Introduction to Data Science and Python](https://github.com/ledaniel2/python-data-science-book/blob/main/01-introduction.md)
2. [Getting Started with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/02-getting-started.md)
3. [Python Libraries for Data Science](https://github.com/ledaniel2/python-data-science-book/blob/main/03-libraries-data-science.md)
4. [Working with Data in Python](https://github.com/ledaniel2/python-data-science-book/blob/main/04-working-data.md)
5. [Data Cleaning and Preprocessing](https://github.com/ledaniel2/python-data-science-book/blob/main/05-data-cleaning.md)
6. [Exploratory Data Analysis (EDA)](https://github.com/ledaniel2/python-data-science-book/blob/main/06-data-analysis.md)
7. [Feature Engineering and Selection](https://github.com/ledaniel2/python-data-science-book/blob/main/07-features.md)
8. [Data Manipulation with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/08-data-manipulation.md)
9. [Advanced Data Visualization](https://github.com/ledaniel2/python-data-science-book/blob/main/09-data-visualization.md)
10. [Model Evaluation and Validation](https://github.com/ledaniel2/python-data-science-book/blob/main/10-model-evaluation.md)
11. [Machine Learning with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/11-machine-learning.md)
12. [Advanced Machine Learning Techniques](https://github.com/ledaniel2/python-data-science-book/blob/main/12-advanced-techniques.md)
13. [Deep Learning with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/13-deep-learning.md)
14. [Time Series Analysis with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/14-time-series-analysis.md)
15. [Natural Language Processing (NLP) with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/15-natural-language.md)
16. [Web Scraping with Python](https://github.com/ledaniel2/python-data-science-book/blob/main/16-web-scraping.md)
17. [Working with APIs in Python](https://github.com/ledaniel2/python-data-science-book/blob/main/17-working-apis.md)
18. [Ethical Considerations in Data Science](https://github.com/ledaniel2/python-data-science-book/blob/main/18-ethics.md)
19. [Data Science Best Practices](https://github.com/ledaniel2/python-data-science-book/blob/main/19-best-practices.md)
20. [Conclusion](https://github.com/ledaniel2/python-data-science-book/blob/main/20-conclusion.md)

All text and code Copyright (c) 2023 Richard Spencer

Released under Creative Commons — CC0 1.0 Universal License
