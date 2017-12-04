# Recursive graph bisection

This program implements the following graph reordering technique:

- Laxman Dhulipala, Igor Kabiljo, Brian Karrer, Giuseppe Ottaviano, Sergey Pupyrev, Alon Shalita:
Compressing Graphs and Indexes with Recursive Graph Bisection. KDD 2016: 1535-1544


## Requirements

The program requires `cilkplus` which should be available in all newish compilers (e.g. gcc7)

## Installing and Compiling

To install and compile the code run the following commands

```
git clone https://github.com/mpetri/recursive_graph_bisection.git
```

And compile using `make`:

```
make
```

which produces the `rec_graph_bisect.x` binary.


## Input format

We use the [ds2i](https://github.com/ot/ds2i) input format (description taken from the repo):


A _binary sequence_ is a sequence of integers prefixed by its length, where both
the sequence integers and the length are written as 32-bit little-endian
unsigned integers.

A _collection_ consists of 3 files, `<basename>.docs`, `<basename>.freqs`,
`<basename>.sizes`.

* `<basename>.docs` starts with a singleton binary sequence where its only
  integer is the number of documents in the collection. It is then followed by
  one binary sequence for each posting list, in order of term-ids. Each posting
  list contains the sequence of document-ids containing the term.

* `<basename>.freqs` is composed of a one binary sequence per posting list, where
  each sequence contains the occurrence counts of the postings, aligned with the
  previous file (note however that this file does not have an additional
  singleton list at its beginning).

* `<basename>.sizes` is composed of a single binary sequence whose length is the
  same as the number of documents in the collection, and the i-th element of the
  sequence is the size (number of terms) of the i-th document.


## Running command

To reorder the index the following options are provided:

`rec_graph_bisect.x <ds2i_prefix> <ds2i_out_prefix> <min_list_len> <num threads>`

where

* `ds2i_prefix` is the `<basename>` specified above

* `ds2i_out_prefix` is the output prefix where the reordered index should be stored

* `min_list_len` specifies a minimum list threshold which should be ignored during reordering. This does not mean the lists will be lost. Lists below the threshold are just not considered in the reordering phase but will still appear in the final output.

* `num_threads` specifies the number of threads to use during computation

## Example

Say you have stored `gov2` in `ds2i` format described above in a directory:


```
[10:56:28 mpetri]$ ls -l /storage/gov2-d2si/
total 43084248
-rw-r--r-- 1 mpetri mpetri 21765004632 Jul 18 14:37 gov2.docs
-rw-r--r-- 1 mpetri mpetri 21765004624 Jul 18 14:37 gov2.freqs
-rw-r--r-- 1 mpetri mpetri    98831272 Jul 18 14:37 gov2.sizes
```

so the `ds2i_prefix` would be `/storage/gov2-ds2i/gov2`

and we execute the bisection command with the parameters

```
rec_graph_bisect.x /storage/gov2-ds2i/gov2 /storage/gov2-ds2i/gov2-bisected 256 32`
```

which uses `32` threads, a minimum list length of `32` and stores the result as:

```
[10:56:28 mpetri]$ ls -l /storage/gov2-d2si/
total 43084248
-rw-r--r-- 1 mpetri mpetri 21765004632 Jul 18 14:37 gov2.docs
-rw-r--r-- 1 mpetri mpetri 21765004624 Jul 18 14:37 gov2.freqs
-rw-r--r-- 1 mpetri mpetri    98831272 Jul 18 14:37 gov2.sizes
-rw-r--r-- 1 mpetri mpetri 21765004632 Jul 18 18:32 gov2-bisected.docs
-rw-r--r-- 1 mpetri mpetri 21765004624 Jul 18 18:32 gov2-bisected.freqs
-rw-r--r-- 1 mpetri mpetri    98831272 Jul 18 18:32 gov2-bisected.sizes
-rw-r--r-- 1 mpetri mpetri    98831272 Jul 18 18:32 gov2-bisected.mapping
```

where `gov2-bisected.mapping` specifies how the document identifiers were remapped in the following format:

```
<initial id> <new id>
```

## Runtime

The code is not as optimized as in the paper but finishes in reasonable time frame. For example, `gov2`
can be reordered in less than two hours.

However, memory consumption is quite high. It requires at least O(size of input files) RAM.

## Authors

* **Matthias Petri** - [mpetri](https://github.com/mpetri)

* **Joel Mackenzie** - [JMMackenzie](https://github.com/JMMackenzie)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details
