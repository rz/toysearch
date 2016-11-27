import glob
import collections
from datetime import datetime
import os
import pickle
import random
import string
import sys
import urllib

import bs4
import numpy


class SearchEngine(object):
    data_path_prefix = 'en'

    def read_articles(self):
        print('reading articles, please wait...')
        t_s = datetime.now()
        self.articles = collections.OrderedDict()
        for pathname in sorted(glob.glob(self.data_path_prefix + '/**', recursive=True)):
            if not os.path.isdir(pathname):
                with open(pathname) as f:
                    try:
                        self.articles[pathname] = f.read()
                    except Exception as e:
                        print(e)
                        print(pathname)
        time_elapsed = datetime.now() - t_s
        print('read articles in {}s'.format(time_elapsed.total_seconds()))


class SimpleSearchEngine(SearchEngine):
    def __init__(self):
        self.read_articles()

    def search(self, query):
        return [article for article, contents in self.articles.items() if query in contents]


class InvertedIndexEngine(SearchEngine):
    index_path = 'inverted_index.pickle'

    def __init__(self):
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.build_index()
            self.save_index()

    def build_index(self):
        print('building index, please wait...')
        t_s = datetime.now()
        self.read_articles()
        index = collections.defaultdict(list)

        punctuation_table = str.maketrans({key: ' ' for key in string.punctuation})
        for article, contents in self.articles.items():
            word_list = contents.translate(punctuation_table).split()
            counter = collections.Counter(word_list)
            for word, count in counter.items():
                index[word].append((article, count/len(word_list)))
        for word, tuples in index.items():
            index[word] = list(set(tuples))
        time_elapsed = datetime.now() - t_s
        print('built index in in {}s'.format(time_elapsed.total_seconds()))
        self.index = index

    def load_index(self):
        print('loading index stored in %s, please wait...' % self.index_path)
        t_s = datetime.now()
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        time_elapsed = datetime.now() - t_s
        print('loaded index in {}s'.format(time_elapsed.total_seconds()))

    def save_index(self):
        print('storing index in %s' % self.index_path)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)

    def search(self, query):
        raise NotImplementedError


class NaiveInvertedIndexEngine(InvertedIndexEngine):
    def search(self, query):
        results = []
        for word in query.split():
            results.extend([w for w, _ in self.index[word]])
        return results


class FrequencyInvertedIndexEngine(InvertedIndexEngine):
    def search(self, query):
        results = []
        for word in query.split():
            results.extend(self.index[word])
        results = [a for a, f in sorted(results, key=lambda t:t[1], reverse=True)]
        unique_results = []
        for r in results:
            if r not in unique_results:
                unique_results.append(r)
        return unique_results


class PageRankIndexEngine(InvertedIndexEngine):
    alpha = 0.85
    iterations = 500
    soup_parser = 'lxml'
    adjacency_dict_path = 'adjacency_dict.pickle'
    smatrix_path = 'smatrix.pickle'
    pagerank_path = 'pagerank.pickle'

    def __init__(self):
        self.read_articles()
        super().__init__()

        if os.path.exists(self.adjacency_dict_path):
            self.load_attr_file(self.adjacency_dict_path, 'adjacency_dict')
        else:
            self.generate_adjacency_dict()
            self.save_attr_file(self.adjacency_dict_path, 'adjacency_dict')

        if os.path.exists(self.smatrix_path):
            self.load_attr_file(self.smatrix_path, 'smatrix')
        else:
            self.generate_smatrix()
            self.save_attr_file(self.smatrix_path, 'smatrix')

        if os.path.exists(self.pagerank_path):
            self.load_attr_file(self.pagerank_path, 'pagerank')
        else:
            self.compute_pagerank()
            self.save_attr_file(self.pagerank_path, 'pagerank')

    def load_attr_file(self, path, attr_name):
        print('loading {} stored in {}, please wait...'.format(attr_name, path))
        t_s = datetime.now()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            setattr(self, attr_name, data)
        time_elapsed = datetime.now() - t_s
        print('loaded {} in {}s'.format(attr_name, time_elapsed.total_seconds()))

    def save_attr_file(self, path, attr_name):
        print('storing {} in {}'.format(attr_name, path))
        attr = getattr(self, attr_name)
        try:
            # this will work on numpy objects
            attr.dump(path)
        except AttributeError:
            print('.dump() failed, pickling manually')
            with open(path, 'wb') as f:
                pickle.dump(attr, f)

    def generate_adjacency_dict(self):
        print('generating adjacency list, this will take several minutes, please wait...')
        t_s = datetime.now()
        self.adjacency_dict = collections.OrderedDict()
        for i, (source_article, contents) in enumerate(self.articles.items()):
            if i % 500 ==0:
                elapsed = datetime.now() - t_s
                print('{:4d} / {} processing {} time elapsed: {}'.format(i, len(self.articles), source_article, elapsed.total_seconds()))
            self.adjacency_dict[source_article] = self.get_outlinks(source_article, contents)

    def get_outlinks(self, article_name, article_contents):
        # return list(set(random.sample(self.articles.keys(), random.randint(0,20)) + ['en/articles/b/r/a/Brazil.html']))
        html_parse = bs4.BeautifulSoup(article_contents, self.soup_parser)

        href_filter = lambda u: u is not None and u.startswith('../../../../articles/')
        atag_to_url = lambda t: 'en/' + t.attrs['href'].rpartition('../')[2].partition('#')[0]

        outlinks = html_parse.find_all('a', attrs={'href': href_filter})
        outlinks = [atag_to_url(l) for l in outlinks]
        return [l for l in outlinks if l in self.articles and l != article_name]

    def generate_smatrix(self):
        print('generating smatrix, please wait...')
        t_s = datetime.now()
        dim = len(self.articles)
        smatrix = numpy.zeros((dim, dim))
        # fill out the values using the adjacency list
        for source_article, contents in self.articles.items():
            outlinks = self.adjacency_dict[source_article]
            for target_article in outlinks:
                source_article_index = self.article_to_index(source_article)
                target_article_index = self.article_to_index(target_article)
                rank = outlinks.count(target_article) / len(outlinks)
                smatrix[source_article_index][target_article_index] = rank

        # make the matrix stochastic by replacing 0 rows with rows with 1/n's
        for i, row in enumerate(smatrix):
            if all([n == 0 for n in row]):
                smatrix[i] = numpy.full((1, dim), 1/dim)

        self.smatrix = smatrix
        time_elapsed = datetime.now() - t_s
        print('generated smatrix in {}s'.format(time_elapsed.total_seconds()))

    def compute_pagerank(self):
        print('computing pagerank, please wait...')
        t_s = datetime.now()
        dim = len(self.articles)
        telematrix = numpy.full((dim, dim), 1/dim)
        gmatrix = self.alpha*self.smatrix + (1 - self.alpha) * telematrix
        pr_vector = numpy.full((1, dim), 1/dim)
        for _ in range(self.iterations):
            pr_vector = numpy.dot(pr_vector, gmatrix)
        self.pagerank = {article: pr_vector[0][self.article_to_index(article)] for article in self.articles.keys()}
        time_elapsed = datetime.now() - t_s
        print('computed pagerank in {}s'.format(time_elapsed.total_seconds()))

    def article_to_index(self, article):
        if not hasattr(self, 'index_cache'):
            self.index_cache = {}
        if article not in self.index_cache:
            self.index_cache[article] = list(self.articles.keys()).index(article)
        return self.index_cache[article]

    def index_to_article(self, i):
        key = list(self.articles.keys())[i]
        return self.articles[key]

    def search(self, query):
        results = []
        for word in query.split():
            results.extend(self.index[word])
        results = [a for a, f in sorted(results, key=lambda t: self.pagerank[t[0]] * t[1], reverse=True)]
        unique_results = []
        for r in results:
            if r not in unique_results:
                unique_results.append(r)
        return unique_results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        engine_name = 'simple'
    else:
        engine_name = sys.argv[1]

    if engine_name == 'simple':
        print('using simple engine')
        engine = SimpleSearchEngine()
    elif engine_name == 'naive_inverted_index':
        print('using naive_inverted_index engine')
        engine = NaiveInvertedIndexEngine()
    elif engine_name == 'frequency_inverted_index':
        print('using frequency_inverted_index engine')
        engine = FrequencyInvertedIndexEngine()
    elif engine_name == 'pagerank_index':
        print('using pagerank_index engine')
        engine = PageRankIndexEngine()
    else:
        print('invalid engine! using simple')
        engine = SimpleSearchEngine()

    while True:
        try:
            user_input = input('[ feeling lucky? ] => ')
        except (EOFError, KeyboardInterrupt):
            sys.exit()

        t_s = datetime.now()
        results = engine.search(user_input)
        time_elapsed = datetime.now() - t_s
        output_str = '  Found {} documents matching {} in {}s, showing the first 10:'.format(
            len(results), user_input, time_elapsed.total_seconds())
        print(output_str, end='\n\n')
        for i, r in enumerate(results[:10]):
            print('  {:2d}. {}'.format(i, r))
        print()


