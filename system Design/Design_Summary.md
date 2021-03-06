[Scalability, Availability & Stability Patterns](http://www.slideshare.net/jboner/scalability-availability-stability-patterns/110-All_operations_in_scope_ofa)
[checkcheckzz Summary](https://github.com/checkcheckzz/system-design-interview)
[cyandtery Summary](https://github.com/cyandterry/Python-Study/blob/master/system_design.md)

[Intro to HDFS](https://www.youtube.com/watch?v=ziqx2hJY8Hg)
mapreduce: algorithm,pattern
hadoop: framework using mapreduce
HDFS: distributed file system

Spark: in cluster computing, not tiles to 2 stage mapreduce, based on HDFS, faster than hadoop

[RAID 0, RAID 1, RAID 5, RAID 10 Explained with Diagrams](http://www.thegeekstuff.com/2010/08/raid-levels-tutorial/)

[Inverted index](https://www.quora.com/Information-Retrieval-What-is-inverted-index)
[Bloom Filter](http://blog.csdn.net/v_july_v/article/details/6685894)
Bloom Filter的这种高效是有一定代价的：在判断一个元素是否属于某个集合时，有可能会把不属于这个集合的元素误认为属于这个集合（false positive）。因此，Bloom Filter不适合那些“零错误”的应用场合

[MVC Model View Control](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller)

[REST vs SOAP](http://searchsoa.techtarget.com/tip/REST-vs-SOAP-How-to-choose-the-best-Web-service)
Both web services: Simple Object Access Protocol (SOAP) and Representational State Transfer (REST)

[双层桶划分]（http://taop.marchtea.com/09.04.html）
适用范围：第k大，中位数，不重复或重复的数字

[Memory stack heap](http://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap)
But Basically,
Process is on Heap memory.
Thread is on Stack memory.
Stack is faster while heap is slower
stackoverflow for stack while heap is for memory leak

[NOSQL Pattern]
Key value store
Run on large number of commodity machines
Data are partitioned and replicated among these machines
Relax the data consistency requirement. (because the CAP theorem proves that you cannot get Consistency, Availability and Partitioning at the the same time)

[Web Search Engine from Sergey and Larry Page](http://infolab.stanford.edu/~backrub/google.html)