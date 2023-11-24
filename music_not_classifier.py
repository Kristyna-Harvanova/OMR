import os
from muscima.io import parse_cropobject_list

# Change this to reflect wherever your MUSCIMA++ data lives
CROPOBJECT_DIR = os.path.join('../data/MUSCIMA++/v1.0/data/cropobjects_manual')

cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

## Symbol Classification
'''
Let’s try to tell apart quarter notes from half notes.
However, notes are recorded as individual primitives in MUSCIMA++, so we need to extract notehead-stem pairs from the data using their relationships. Quarter notes are all full-notehead-stem pairs with no beam or flag. Half-notes are all empty-notehead-stem pairs.
After we extract the note classes, we will need to compute features for classification. To do that, we first need to “draw” the symbols in the appropriate relative positions. Then, we can extract whatever features we need.
Finally, we train a classifier and evaluate it.
'''

## Extracting notes and symbols

# Bear in mind that the outlinks are integers, only valid within the same document.
# Therefore, we define a function per-document, not per-dataset.

def extract_notes_from_doc(cropobjects):
    """Finds all ``(full-notehead, stem)`` pairs that form quarter or half notes. 
    Finds also all signle oblects, that are flat, sharp or natural signs.
    Returns five lists.
        Two lists of CropObject tuples: one for quarter notes, one of half notes.
        Three lists of CropObjects: one for flat signs, one for sharp signs, one for natural signs.

    :returns: quarter_notes, half_notes, flats, sharps, naturals
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    flats = []
    sharps = []
    naturals = []

    for c in cropobjects:
        if (c.clsname == 'notehead-full') or (c.clsname == 'notehead-empty'):
            _has_stem = False
            _has_beam_or_flag = False
            stem_obj = None
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == 'stem':
                    _has_stem = True
                    stem_obj = _o_obj
                elif _o_obj.clsname == 'beam':
                    _has_beam_or_flag = True
                elif _o_obj.clsname.endswith('flag'):
                    _has_beam_or_flag = True
            if _has_stem and (not _has_beam_or_flag):
                # We also need to check against quarter-note chords.
                # Stems only have inlinks from noteheads, so checking
                # for multiple inlinks will do the trick.
                if len(stem_obj.inlinks) == 1:
                    notes.append((c, stem_obj))
        
        if (c.clsname == 'flat'):
            flats.append((c,))
        
        if (c.clsname == 'sharp'):
            sharps.append((c,))
        
        if (c.clsname == 'natural'):
            naturals.append((c,))

    quarter_notes = [(n, s) for n, s in notes if n.clsname == 'notehead-full']
    half_notes = [(n, s) for n, s in notes if n.clsname == 'notehead-empty']
    
    return quarter_notes, half_notes, flats, sharps, naturals

all_symbols = [extract_notes_from_doc(cropobjects) for cropobjects in docs]

import itertools
qns = list(itertools.chain(*[qn for qn, hn, f, s, n in all_symbols]))
hns = list(itertools.chain(*[hn for qn, hn, f, s, n in all_symbols]))
fs = list(itertools.chain(*[f for qn, hn, f, s, n in all_symbols]))
ss = list(itertools.chain(*[s for qn, hn, f, s, n in all_symbols]))
ns = list(itertools.chain(*[n for qn, hn, f, s, n in all_symbols]))

#print(len(qns), len(hns), len(fs), len(ss), len(ns))

## Creating note images
'''
Each notehead and stem CropObject has its own mask and its bounding box coordinates. 
We need to combine these two things, in order to create a binary image of the note.
'''

import numpy

def get_image(cropobjects, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
    There will be a given margin of background on the edges."""

    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])

    # Create the canvas onto which the masks will be pasted
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    canvas = numpy.zeros((height, width), dtype='uint8')

    for c in cropobjects:
        # Get coordinates of upper left corner of the CropObject
        # relative to the canvas
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        # We have to add the mask, so as not to overwrite
        # previous nonzeros when symbol bounding boxes overlap.
        canvas[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask

    canvas[canvas > 0] = 1
    return canvas

qn_images = [get_image(qn) for qn in qns]
hn_images = [get_image(hn) for hn in hns]
f_images = [get_image(f) for f in fs]
s_images = [get_image(s) for s in ss]
n_images = [get_image(n) for n in ns]

'''Let’s visualize some of these notes, to check whether everything worked. 
(For this, we assume you have matplotlib. If not, you can skip this step.)
'''

import matplotlib.pyplot as plt

def show_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.show()

def show_masks(masks, row_length=5):
    n_masks = len(masks)
    n_rows = n_masks // row_length + 1
    n_cols = min(n_masks, row_length)
    fig = plt.figure()
    for i, mask in enumerate(masks):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(mask, cmap='gray', interpolation='nearest')
    # Let's remove the axis labels, they clutter the image.
    for ax in fig.axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()

# show_masks(qn_images[:25])
# show_masks(hn_images[:25])
# show_masks(f_images[:25])
# show_masks(s_images[:25])
# show_masks(n_images[:25])

## Feature Extraction
'''
Now, we need to somehow turn the note images into classifier inputs.
Let’s get some inspiration from the setup of the HOMUS dataset. In their baseline classification experiments, 
the authors just resized their images to 20x20. For notes, however, this may not be such a good idea, 
because it will make them too short. Let’s instead resize to 40x10.
'''

from skimage.transform import resize

qn_resized = [resize(qn, (40, 10)) for qn in qn_images]
hn_resized = [resize(hn, (40, 10)) for hn in hn_images]
f_resized = [resize(f, (40, 10)) for f in f_images]
s_resized = [resize(s, (40, 10)) for s in s_images]
n_resized = [resize(n, (40, 10)) for n in n_images]

# And re-binarize, to compensate for interpolation effects
for qn in qn_resized:
    qn[qn > 0] = 1
for hn in hn_resized:
    hn[hn > 0] = 1
for f in f_resized:
    f[f > 0] = 1
for s in s_resized:
    s[s > 0] = 1
for n in n_resized:
    n[n > 0] = 1

# show_masks(qn_resized[:25])
# show_masks(hn_resized[-25:])
# show_masks(f_resized[:25])
# show_masks(s_resized[:25])
# show_masks(n_resized[:25])

## Classification
'''
We now need to add the output labels and make a train-dev-test split out of this.
Let’s make a balanced dataset, to keep things simpler.
'''

# Find the minimum count among all categories to balance the dataset size
min_count = min(len(hn_resized), len(qn_resized), len(f_resized), len(s_resized), len(n_resized))
# Randomly sample symbols from each category to match the minimum count
import random
random.shuffle(hn_resized)
hn_selected = hn_resized[:min_count]
random.shuffle(qn_resized)
qn_selected = qn_resized[:min_count]
random.shuffle(f_resized)
f_selected = f_resized[:min_count]
random.shuffle(s_resized)
s_selected = s_resized[:min_count]
random.shuffle(n_resized)
n_selected = n_resized[:min_count]

# Now, create the output labels and merge the data into one dataset.
N_LABEL = 4
S_LABEL = 3
F_LABEL = 2
Q_LABEL = 1
H_LABEL = 0

n_labels = [N_LABEL for _ in n_selected]
s_labels = [S_LABEL for _ in s_selected]
f_labels = [F_LABEL for _ in f_selected]
qn_labels = [Q_LABEL for _ in qn_selected]
hn_labels = [H_LABEL for _ in hn_selected]

symbols = hn_selected + qn_selected + f_selected + s_selected + n_selected
# Flatten data
symbols_flattened = [n.flatten() for n in symbols]
labels = hn_labels + qn_labels + f_labels + s_labels + n_labels

# Let’s use the sklearn package for experimental setup. Normally, 
# we would do cross-validation on data of this small size, but for the purposes of the tutorial, 
# we will stick to just one train/test split.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    symbols_flattened, labels, test_size=0.25, random_state=42,
    stratify=labels)

# What could we use to classify this data? Perhaps a k-NN classifier might work.
from sklearn.neighbors import KNeighborsClassifier

K=5

# Trying the defaults first.
clf = KNeighborsClassifier(n_neighbors=K)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
clf.fit(X_train, y_train)

# Let’s run the classifier now and evaluate the results.
y_test_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred, target_names=['half', 'quarter', 'flat', 'sharp', 'natural']))