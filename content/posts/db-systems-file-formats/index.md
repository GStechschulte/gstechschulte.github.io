+++
title = 'Database Systems - Storage'
date = 2024-12-01
author = 'Gabriel Stechschulte'
categories = ['database-systems']
draft = true
+++

## Introduction

As the business landscape embraces data-driven approaches for analysis and decision-making, there is a rapid surge in the volume of data requiring storage and processing. This surge has led to the growing popularity of OLAP database systems.

An OLAP system workload is characterized by complex queries that require scanning over large portions of the database. In OLAP workloads, the database system is often analyzing and deriving new data from existing data collected on the OLTP side. In contrast, OLTP workloads are characterized by fast, relatively simple and repetitive queries that operate on a single entity at a time (usually involving an update or insert).

This blog aims to provide an overview of the popular data storage representations and encoding within OLTP and OLAP database systems. First, OLTP data storage is discussed followed by OLAP systems.

## Storage models

The difference in access patterns between OLTP and OLAP means that each system can optimize for their respective data access patterns, e.g. OLAP systems can optimize for techniques like sequential scans where the system can scan through large chunks of data at a time. Due to the different access patterns, OLTP and OLAP systems have different storage models. A DBMS's storage model specifies how it physically organizes tuples on disk and in memory. There are three primary storage models: (1) N-ary (row-oriented), (2) decomposition (columnar), and (3) hybrid. Each of these storage models is discussed below.

### Storage manager

Assuming the DB is not in-memory, a DBMS stores a DB as files on disk. The DBMS _storage manager_ is responsible for managing the DB's files, e.g. keeping track of what has been read and written to pages as well as how much free space is in these pages. It represents these files as a collection of pages



### Row-oriented

Assuming the DB is not in-memory, a DBMS stores a DB as files on disk. The DBMS _storage manager_ is responsible for managing the DB's files, e.g. keeping track of what has been read and written to pages as well as how much free space is in these pages. It represents these files as a collection of pages and in row-oriented storage, the DBMS stores (almost) all the attributes for a single tuple (row) contiguously in a single page in a file. This is ideal for OLTP workloads because transactions usually access individual entities and are insert-heavy.

At a high level, the DBMS is managing a bunch of files. Within these files there are pages to break up the data into different chunks. Then, within this page, theere are tuples, i.e. the data (rows/records) of the tables.

#### Pages

The DBMS organizes the DB across one or more files in fixed-size blocks called _pages_. Pages contain different kinds of data such as tuples, indexes, and log records. Most systems will not mix these types within a page. Additionally, some systems require that pages are _self-contained_, i.e. all the information needed to read each page is on the page itself.

Each page is given a unique identifier, and most DBMSs have an indirection layer that maps a page id to a file path and offset. The upper levels of the system will ask for a specific page number. Then, the storage manager will have to turn that page number into a file and an offset to find the page.

**TODO: insert diagram here**

#### Page storage architecture

**TODO: content**

#### Page layout

Every page contains a header of metadata about the pages's:
- Page size
- Checksum
- DBMS version
- Transaction visibility

**TODO: insert diagram here**

There are three main approaches to organizing data within a page: (1) slotted (tuple-oriented), (2) log-structured, and (3) index-organized.

##### Slotted pages

Slotted pages is the most common approach for row-oriented DBMSs for laying out data within pages. Slotted pages map slots to offsets. A header keeps track of the number of slots used with an offset of the starting location of the last used slot, and a slot array, which keeps track of the location of the start of each tuple. To add a tuple, the slot array will grow from the beginning to the end, and the data of the tuples will grow from end to beginning. The page is considered full when the slot array and the tuple data meet.

**TODO: insert diagram here**

##### Tuple layout

The DBMS assigns each logical tuple a unique identifier that represents its physical location in the DB. A tuple is essentially a sequence of bytes and it is the DBMSs job to interpret those bytes into attribute types and values. Typically, the layout of a tuple consists of: a header, the data, unique identifier, and optionally denormalized tuple data.

**TODO: insert diagram here**

- **Tuple header**. Each tuple is prefixed with a header that contains metadata about it such as visibility information for the DBMS's concurrency control protocol, and a bit map for NULL values.
- **Tuple data**. The actual data for attributes. Attributes are typically stored in the order that you specify them when the table is created. Most DBMSs do not allow a tuple to exceed the size of a page.
- **Unique identifer**. Each tuple in the DB is assigned a unique identifier. Most commonly this is `page_id + (offset or slot`.
- **Denormalized tuple data**. A DBMS can physically denormalize (e.g. pre-join) related tuples and store them together in the same page. This makes reads faster since the DBMS only has to load one page rather than two separate pages. However, it can make updates more expensive since the DBMS needs more space for each tuple.

**TODO: insert diagram here**

#### Summary

A DB is organized as files on disk. These files are composed of pages. There are multiple ways to organize data within these pages (e.g. slotted). The data we care about is stored as tuples within the pages.

### Column-oriented

OLAP workloads typically require scanning over large portions of a table using a few columns (relative to the total number of columns) to analyze data. File sizes in OLAP workloads are relatively large (usually 100MB+) and are primarily read-only. Thus, the DBMS should store a single attribute for all tuples contiguously in a block of data in memory for column-oriented storage. Although the file sizes are "large", the DBMS may still organize data into groups with the file.

**TODO: insert diagram here**

Moreover, for column-oriented storage, all variable length data needs to be converted to fixed length so that simple arithmetic can be used to jump to an offset to find a tuple. This can be done by using dictionary compression. The DBMS stores the dictionary in the header of the page and stores the actual data in the body of the page. The DBMS can then use the dictionary to reconstruct the data. To identify the tuples (data in page), there are two primary options:

1. **Fixed length offsets**. Each value is the same length for an attribute. The DBMS can reach locations of other attributes of the same tuple by inferring from the length of the value and the current offset.

**TODO: insert diagram here**

2. **Embedded tuple ids**. Each value is stored with its tuple ID in a column.

**TODO: insert diagram here**

Furthermore, OLAP queries rarely select a single column, i.e. the projection and predicates often involve different columns. For example:

```sql
SELECT product_id,
	AVG(price)
FROM sales
WHERE time > '2024-01-01'
GROUP BY product_id;
```
Thus, a columnar scheme that still stores attributes separately but keeps the data for each tuple physically close to each other is desired.

### Hybrid (PAX)

Partition across attributes (PAX) is a hybrid storage model that horizontally partitions data into _row groups_. Then, vertically partitions thir attributes into _column chunks_. All within a DB page. This is what Parquet and Orc use. The goal with PAX is to get the benefit of faster processing on columnar storage while retaining the spatial locality benefits of row storage.

**TODO: insert diagram here**

In most PAX model implementations such as Apache Parquet and Orc, the global metadata is the footer of the file. This is because most distributed file systems and OLAP workloads are very append-friendly and may not support in-place updates efficiently.

#### Summary

**TODO: add content**

### Format design decisions

Modern row-oriented and columnar systems need to make certain design decisions when designing and engineering file formats. Here, the major design decisions behind file formats for OLAP workloads are discussed:

- File metadata
- Format layout
- Type system
- Encoding schemes
- Block compression
- Filters
- Nested data

#### File metadata

Files are self-contained to increase portability, i.e. they contain all the relevant information to interpret their contents without external data dependencies. Each file maintains global metadata (usually in the footer) abouts it contents such as: table schema, row group offsets, and tuple/zone counts.

This is opposite of, for example, Postgres. In Postgres, you have a bunch of files that keep track of the catalog (schema, tables, types, etc.). Then, you have pages for the actual data. In order for you to understand what is in the data pages, you need to read the catalog first.

#### Format layout

The most common file formats like Parquet and Orc use the PAX storage model that splits data into row groups that contain one or more column chunks. However, the size of row groups varies per implementation and makes compute/memory trade offs.

- **Parquet**. Number of tuples (e.g. 1 million)
- **Orc**. Physical storage size (e.g. 250MB)
- **Arrow**. Number of tuples (e.g. 1020 * 1024)

**TODO: insert PAX or Parquet diagram here**

#### Type system

The type system defines the data types that the file format supports. A DB system typically has both physical and logical types.

- **Physical type**. Low-level byte representation, e.g. IEEE-754, that focuses on the actual storage representation.

- **Logical type**. Auxiliary types are higher-level representation that focus on the semantic meaning, e.g. `DATE`, `INT64`, `VARCHAR`, that are then mapped to physical types.

#### Encoding schemes

Encoding schemes specify how the file format stores the bytes for contiguous data. There are several encoding schemes (given below), and one can apply multiple encoding schemes on top of each other to further improve compression).

**Dictionary encoding**. The most common. It replaces frequent values with smaller fixed-length codes and then maintains a mapping (dictionary) from the codes to the original values. Codes could either be positions (using a hash table) or byte offsets into a dictionary. Additionally, values in the dictionary can be sorted and compressed further.

**TODO: insert dictionary encoding diagram here**

**Run-length encoding (RLE)**. <content>

**Bitpacking**. <content>

**Delta encoding**. <content>

**Frame-of-reference (FOR)**. <content>

#### Compression

Compression compresses data using a general-purpose algorithm (e.g. LZO, LZ4, Snappy, Zstd) and saves storage space, but can introduce computational overhead (compress versus decompress) and data opacity for the execution engine. Data opacity means if you run data through Snappy or Zstd, the DB system does not know what those bytes mean and you cannot go and jump to arbitrary offsets to find the data you are looking for. You need to decompress the whole block to interpret the data.

Compression made more sense in the 2000s and 2010s because the main bottleneck was disk and network, so we were willing to pay CPU costs. But now, the CPU is actually one of the slower components and we have cheap object stores.

#### Filters

First, the difference between a filter and an index. An index tells you were data is, whereas a filter tells you if something does exist. There are several types of filters in DB systems to boost search performance.

**Zone maps**. Maintain min/max values per column at the file and row group level. Parquet and Orc store zone maps in the header of each row group.

**Bloom filters**. A probabilistic data structure (can get false positives but never false negatives) that tracks the existence of values for each column in a row group.

#### Nested data

Real-world data sets often contain semi-structured objects, e.g. JSON and Protobufs. In order to store semi-structured data as regular columns, most modern formats add additional fields that make querying the data easier and faster. There are two main approaches to storing semi-structured data as columns: (1) record shredding, and (2) length + presence encoding.

**Record shredding**. When storing semi-structured data as a "blob" in a column, every single time you need to parse the blob, you need to run JSON functions to extract the structure from it. Instead, split it up so that every level in the path is treated as a separate column. Now we can rip through a column for a given field in the JSON. There is always a schema! It does not make sense to have applications inserting random documents into a table!

To achieve this, two additional fields are stored:

- **Repetition level**. At what repeated field in the field's path the value has repeated, i.e. for each path store it as a separate column and record how many steps deep we are into a given document for that hierarchy. Essentially, we are storing paths as separate columns with additional metadata about the paths.
- **Definition level**. Specifies how many columns in the path of the field that could be undefined are actually present.

**TODO: insert shreding diagram here**

**Length + presence encoding**.

#### Summary
