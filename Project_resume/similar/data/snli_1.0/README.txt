This is the 1.0 distribution of the Stanford Natural Language Inferece (SNLI) Corpus.

If you use this corpus, please cite this paper: 

	http://nlp.stanford.edu/pubs/snli_paper.pdf

	Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015.
	A large annotated corpus for learning natural language inference. 
	Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).

	@inproceedings{snli:emnlp2015,
		Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher, and Manning, Christopher D.},
		Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
		Publisher = {Association for Computational Linguistics},
		Title = {A large annotated corpus for learning natural language inference},
		Year = {2015}}

Contact: sbowman@stanford.edu

Project page: http://nlp.stanford.edu/projects/snli/


== Fields ==

sentence1: The premise caption that was supplied to the author of the pair.

sentence2: The hypothesis caption that was written by the author of the pair.

sentence{1,2}_parse: The parse produced by the Stanford Parser (3.5.2, case insensitive PCFG, trained on the standard training set augmented with the parsed Brown Corpus) in Penn Treebank format.

sentence{1,2}_binary_parse: The same parse as in sentence{1,2}_parse, but formatted for use in tree-structured neural networks with no unary nodes and no labels.

annotator_labels (label1-5 in the tab separated file): These are all of the individual labels from annotators in phases 1 and 2. The first label comes from the phase 1 author, and is the only label for examples that did not undergo phase 2 annotation. In a few cases, the one of the phase 2 labels may be blank, indicating that an annotator saw the example but could not annotate it.

gold_label: This is the label chosen by the majority of annotators. Where no majority exists, this is '-', and the pair should not be included when evaluating hard classification accuracy.

captionID: A unique identifier for each sentence1 from the original Flickr30k example.

pairID: A unique identifier for each sentence1--sentence2 pair.

NOTE: captionID and pairID contain information that can be useful in making classification decisions and should not be included in model input (nor, of course, should either annotator_labels or gold_label).


== Data sources ==

The sentence1 sentences drawn from two sources. Except where otherwise marked, they are drawn from the Flickr30k corpus and are licenced as CreativeCommons Attribution-ShareAlike and can be cited as:

	Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. 
	From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. 
	Transactions of the Association for Computational Linguistics 2: 67--78.

	@article{young:tacl2014,
		Author = {Young, Peter and Lai, Alice and Hodosh, Micah and Hockenmaier, Julia},
		Date-Added = {2015-05-29 01:51:07 +0000},
		Date-Modified = {2015-08-17 15:42:39 +0000},
		Journal = {Transactions of the Association for Computational Linguistics},
		Pages = {67--78},
		Title = {From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
		Volume = {2},
		Year = {2014}}

About 4k sentences in the training set have captionIDs and pairIDs beginning with `vg_'. These come from a pilot data collection effort that used data from the VisualGenome corpus, which is still under construction as of the release of SNLI. For more information on VisualGenome, see: https://visualgenome.org/


== Staistics ==

Training pairs: 550152
Dev pairs: 10000
Test pairs: 10000
Total pairs: 570152

Train labels: {'entailment': 183416, 'neutral': 182764, '-': 785, 'contradiction': 183187}
Dev labels: {'entailment': 3329, 'neutral': 3235, '-': 158, 'contradiction': 3278}
Test labels: {'entailment': 3368, 'neutral': 3219, '-': 176, 'contradiction': 3237}
LHS sentences parsed with root label S: 0.739764133073
RHS sentences parsed with root label S: 0.888862969875

Mean tokens per LHS sentence (averaging over pairs): 14.0717791045
Mean tokens per RHS sentence (averaging over pairs): 8.25575811363

LHS length distribution: {2: 42, 3: 156, 4: 1095, 5: 3882, 6: 12120, 7: 26514, 8: 37434, 9: 44028, 10: 49245, 11: 50919, 12: 48363, 13: 43314, 14: 38119, 15: 33183, 16: 27619, 17: 23250, 18: 20247, 19: 18513, 20: 16386, 21: 13746, 22: 12066, 23: 9183, 24: 7131, 25: 6198, 26: 5007, 27: 3963, 28: 3438, 29: 2631, 30: 1959, 31: 1956, 32: 1434, 33: 1086, 34: 912, 35: 897, 36: 774, 37: 453, 38: 618, 39: 291, 40: 330, 41: 249, 42: 180, 43: 225, 44: 162, 45: 108, 46: 87, 47: 60, 48: 36, 49: 90, 50: 21, 51: 66, 52: 51, 53: 36, 54: 24, 55: 63, 56: 18, 57: 15, 58: 6, 59: 27, 60: 6, 61: 3, 62: 3, 63: 3, 64: 6, 65: 3, 66: 3, 67: 6, 68: 6, 69: 18, 70: 15, 71: 3, 72: 15, 73: 3, 75: 15, 79: 3, 82: 15}

RHS length distribution: {0: 38, 2: 1011, 3: 7980, 4: 29471, 5: 61196, 6: 74094, 7: 93600, 8: 85848, 9: 61359, 10: 46712, 11: 33241, 12: 22843, 13: 15994, 14: 11047, 15: 7601, 16: 5312, 17: 3731, 18: 2631, 19: 1878, 20: 1325, 21: 911, 22: 642, 23: 449, 24: 357, 25: 217, 26: 168, 27: 138, 28: 84, 29: 67, 30: 46, 31: 26, 32: 31, 33: 23, 34: 16, 35: 19, 36: 8, 37: 12, 38: 4, 39: 5, 40: 2, 41: 4, 42: 2, 43: 1, 44: 1, 48: 1, 51: 1, 55: 2, 56: 1, 60: 1, 62: 1}

Total validated pairs (4x each, plus the original annotation): 56951 a.k.a. 0.0998873984481
Fraction where gold != author label or no gold label: 0.0874962687222
Fraction where gold != author label: 0.0678477989851
Fraction with no gold label: 0.0196484697371
Fraction with unanimous agreement: 0.582588541026
% annotator agreement with gold labels: 0.889849081482
% annotator agreement with original labels: 0.857684676301
