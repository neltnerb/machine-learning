import tensorflow as tf
from rdkit import Chem
import csv
import itertools as it
import numpy as np
from pandas import *
import networkx as nx
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

# First I need to import the raw data from the csv file.
#
# Note, I modified the original solubility.txt file to manually replace
# commas in compound names with semicolons so that the extractor doesn't
# confuse them for delimiters. I simultaneously parse the SMILES string
# into a rdkit molecule object for easier feature extraction.

y_data = []
SMILES_data = []
mol_data = []

# atom_data will contain a list of example molecules, with each member
# list indexed by the atom index and then with a number representing
# the atomic number of that atom.

ifile = open('solubility.txt', "rb")
reader = csv.reader(ifile, delimiter=',')

# This is the best way I could find to skip the first iterator. The
# method using islice requires that you know how many elements are in
# the iterator, which is definitely more complex than just using a
# flag.

isheader = True
maxAtoms = 0

for row in reader:
    if isheader:
        isheader = False
    else:
        y_data.append(float(row[1]))
        SMILES_data.append(row[3])
        molecule = Chem.MolFromSmiles(row[3])
        numAtoms = molecule.GetNumAtoms()
        if numAtoms > maxAtoms:
            maxAtoms = numAtoms
        mol_data.append(molecule)
        
# Manually set maxAtoms to help get the TensorFlow code right.
maxAtoms = 60
        
# Create numpy array for atom descriptors. Size is [n, i, m] where n
# is the number of examples, i is the max number of atoms in a molecule
# seen while parsing the data, and m is 11 to represent H, C, N, O, F,
# P, S, Cl, Br, I, or metal.

atom_data = np.zeros((len(mol_data), maxAtoms, 11), dtype = float)

# Create numpy array for bond descriptors. Size is [n, i, i, m] where n
# is the number of examples, i is the max number of atoms in a molecule,
# and m is 4 to represent single, double, triple, and aromatic bonds.

bond_data = np.zeros((len(mol_data), maxAtoms, maxAtoms, 4), dtype = float)

# Create numpy array for graph distance. Size is [n, i, i, 7] where n
# is the number of examples, i is the max number of atoms in a molecule,
# and m is 7 where the boolean array contains True if the distance from
# the first atom to the second is less than the bin index. So,
# [0, 0, 1, 3] means whether in training example 0, atoms 0 and 1 are
# closer than 3 bonds away at the shortest distance. These will be
# calculated by building an adjacency matrix from the bond_data array
# and using networkx to convert that to a graph representation which
# is easier to extract the shortest path lengths from.

example_index = 0
distance_data = np.zeros((len(mol_data), maxAtoms, maxAtoms, 7), dtype = float)
graph_data = []

for molecule in mol_data:
    for atom in molecule.GetAtoms():
        atomicNum = atom.GetAtomicNum()
        if atomicNum == 1:
            atom_data[example_index,atom.GetIdx(),0] = 1
        elif atomicNum == 6:
            atom_data[example_index,atom.GetIdx(),1] = 1
        elif atomicNum == 7:
            atom_data[example_index,atom.GetIdx(),2] = 1
        elif atomicNum == 8:
            atom_data[example_index,atom.GetIdx(),3] = 1
        elif atomicNum == 9:
            atom_data[example_index,atom.GetIdx(),4] = 1
        elif atomicNum == 15:
            atom_data[example_index,atom.GetIdx(),5] = 1
        elif atomicNum == 16:
            atom_data[example_index,atom.GetIdx(),6] = 1
        elif atomicNum == 17:
            atom_data[example_index,atom.GetIdx(),7] = 1
        elif atomicNum == 35:
            atom_data[example_index,atom.GetIdx(),8] = 1
        elif atomicNum == 53:
            atom_data[example_index,atom.GetIdx(),9] = 1
        else:
            atom_data[example_index,atom.GetIdx(),10] = 1
    adjacency = np.zeros((molecule.GetNumAtoms(), molecule.GetNumAtoms()), dtype = int)
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bondtype = bond.GetBondTypeAsDouble()
        if bondtype == 1:
            bond_data[example_index,atom1,atom2,0] = 1
            bond_data[example_index,atom2,atom1,0] = 1
        elif bondtype == 1.5:
            bond_data[example_index,atom1,atom2,3] = 1
            bond_data[example_index,atom2,atom1,3] = 1
        elif bondtype == 2:
            bond_data[example_index,atom1,atom2,1] = 1
            bond_data[example_index,atom2,atom1,1] = 1
        elif bondtype == 3:
            bond_data[example_index,atom1,atom2,2] = 1
            bond_data[example_index,atom2,atom1,2] = 1
    for atom1 in range(0, molecule.GetNumAtoms()):
        for atom2 in range(atom1+1, molecule.GetNumAtoms()):
            if any(bond_data[example_index,atom1,atom2]):
                adjacency[atom1,atom2] = 1
                adjacency[atom2,atom1] = 1
    graph = nx.from_numpy_matrix(adjacency)
    graph_data.append(graph)
    for atom1 in range(0, molecule.GetNumAtoms()):
        lengths = nx.single_source_shortest_path_length(graph, atom1)
        for atom2 in range(atom1, molecule.GetNumAtoms()):
            distance = lengths[atom2]
            for threshold in range(0,7):
                if distance <= threshold:
                    distance_data[example_index,atom1,atom2,threshold] = 1
                    distance_data[example_index,atom2,atom1,threshold] = 1
    example_index += 1

# Uncomment below for checking data validity.

#while 1:
#    print "Which example to query?"
#    example = int(raw_input())
    
#    np.random.shuffle(X)
#    print atom_data
#    print DataFrame(atom_data[example,:10])
#    print DataFrame(bond_data[example,0,:10])
#    print DataFrame(distance_data[example,0,:,:])
#    nx.draw_networkx(graph_data[example])
#    plt.show()
    
# The actual machine learning part. First I will figure out how to implement
# the "weave" layers to fingerprint the molecules based on the above data.

# First step seems to be to convert the numpy arrays I created above into
# TensorFlow tensors. This seems to be okay, but I don't know how to test
# it other than knowing that it doesn't produce errors. The data is 4D and
# large, so viewing it would be quite hard.

atom_train = tf.constant(atom_data[0:600])
bond_train = tf.constant(bond_data[0:600])
distance_train = tf.constant(distance_data[0:600])
y_train = tf.constant(y_data[0:600])

atom_test = tf.constant(atom_data[600:1000])
bond_test = tf.constant(bond_data[600:1000])
distance_test = tf.constant(distance_data[600:1000])
y_test = tf.constant(y_data[600:1000])

# And create flat versions of the features.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float64, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

# Here I'm creating the dummy variables that the feed_dict will replace
# with real data during the optimization. The shapes match the size of
# the input arrays from numpy above.

atoms = tf.placeholder(tf.float64, shape=[None, 60, 11])
bonds = tf.placeholder(tf.float64, shape=[None, 60, 60, 4])
distances = tf.placeholder(tf.float64, shape=[None, 60, 60, 7])
y_ = tf.placeholder(tf.float64, shape=[None])

# Okay, so I created my placeholders, and now I need to tell it how
# to reshape them into a flat representation.

atoms_flat = tf.reshape(atoms, [-1, 60 * 11])
bonds_flat = tf.reshape(bonds, [-1, 60 * 60 * 4])
distances_flat = tf.reshape(distances, [-1, 60 * 60 * 7])

# And then concatenate the columns, followed by another reshape such
# that the total length is now the total number of input features.

x = tf.concat(1, [atoms_flat, bonds_flat, distances_flat])
x_flat = tf.reshape(x, [-1, 60 * 11 + 60 * 60 * 4 + 60 * 60 * 7])

w1 = weight_variable([60 * 11 + 60 * 60 * 4 + 60 * 60 * 7, 200])
b1 = bias_variable([200])
h1 = tf.nn.sigmoid(tf.matmul(x_flat, w1) + b1)

w2 = weight_variable([200, 200])
b2 = bias_variable([200])
h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

w3 = weight_variable([200, 200])
b3 = bias_variable([200])
h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

w4 = weight_variable([200, 200])
b4 = bias_variable([200])
h4 = tf.nn.sigmoid(tf.matmul(h3, w4) + b4)

w5 = weight_variable([200, 1])
b5 = bias_variable([1])
y = tf.matmul(h4, w5) + b5

cost = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(y, y_))))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

sess.run(tf.global_variables_initializer())

for i in range(100):
    train_step.run(feed_dict={atoms: atom_train.eval(), bonds: bond_train.eval(), distances: distance_train.eval(), y_: y_train.eval()})
    train_cost = cost.eval(feed_dict={atoms: atom_train.eval(), bonds: bond_train.eval(), distances: distance_train.eval(), y_: y_train.eval()})
    print("Step %d, Training Cost: %g"%(i, train_cost))

test_cost = cost.eval(feed_dict={atoms: atom_test.eval(), bonds: bond_test.eval(), distances: distance_test.eval(), y_: y_test.eval()})
print("Test Cost: %g"%(test_cost))

actual = np.array(y_test.eval())
predicted = np.array(y.eval(feed_dict={atoms: atom_test.eval(), bonds: bond_test.eval(), distances: distance_test.eval(), y_: y_test.eval()}))

print np.shape(actual)
print np.shape(predicted)

print DataFrame([actual, predicted])
