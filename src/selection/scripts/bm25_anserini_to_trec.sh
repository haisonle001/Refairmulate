# Load Java environment
echo "Running on: $JAVA_HOME"
# export JAVA_HOME=/mnt/data/son/usr/lib/jvm/java-16-openjdk-amd64
# export PATH=$JAVA_HOME/bin:$PATH

# Debugging info
echo "Using Java version:"
java -version

# Run Anserini search directly with Java
java -cp `ls /anserini/target/*-fatjar.jar` -Xms512M -Xmx192G --add-modules jdk.incubator.vector \
    io.anserini.search.SearchCollection \
    -index /anserini/indexes/lucene-index-msmarco \
    -topics /msmarco-query-reformulation/datasets/queries/Gold_target.tsv \
    -topicreader TsvInt \
    -output /data/msmarco/trec/run.neutral_queries.trec \
    -format msmarco \
    -parallelism 128 \
    -bm25 -bm25.k1 0.82 -bm25.b 0.68 -hits 1000

echo "✅ Anserini search completed."

# INPUT_FILE="run.neutral_queries.trec"
# OUTPUT_FILE="run.new_neutral_queries.trec"

# python /mnt/data/son/Thesis/src/anserini_to_trec.py -i $INPUT_FILE -o $OUTPUT_FILE