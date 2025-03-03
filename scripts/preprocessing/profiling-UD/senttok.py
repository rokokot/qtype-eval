# -*- coding: utf-8  -*-
# !/usr/bin/python


class Token:
    """ Class containing an annotated token

    Attributes:
        id (int): token id
        form (unicode): form
        lemma (unicode): lemma
        upos (unicode): UDT pos
        xpos (unicode): language-specific pos
        mfeats (unicode): morphological features
        head (int): head
        dep (unicode): dependency label
        children ([token]): list of children
    """

    def __init__(self, items):
        #print (items)
        self.id = int(items[0])
        self.form = items[1]
        self.lemma = items[2]
        self.upos = items[3]  # universal POS
        self.xpos = items[4]  # language-spec POS
        self.mfeats = items[5]  # morphological features
        self.head = int(items[6])  # head of the current word (if 0 --> root of the whole tree)
        self.dep = items[7]  # UD relation to the HEAD
        self.children = []  # Children of ROOT token (array filled in Sentence())

    def __repr__(self):
        return '\t'.join([str(self.id), self.form,
                          self.lemma, self.upos,
                          self.xpos, self.mfeats,
                          str(self.head), self.dep])


class Sentence:
    """ Class containing annotated sentence

    Attributes:
        tokens ([Token]): the list of tokens
        root (Token): contain the root token which contains all the syntactic tree
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.root = tokens[0]
        for n in tokens:
            if n.head == 0:
                self.root = n
            if n.head != 0 and n.upos != 'PUNCT':
                tokens[n.head - 1].children.append(n)

    def __repr__(self):
        return '\n'.join([str(x) for x in self.tokens])
