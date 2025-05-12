#!/usr/bin/env python
# coding: utf-8

# # 1. Environment Setup

# ### 1.1 Download Dataset

# ### 1.2 Init Spark and Load Dataset

# In[29]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

import numpy as np
import random
from pyspark.sql import Row
from pyspark.sql.functions import col, collect_list, struct, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, StructType, StructField
import math


# In[30]:


path="df_Mid_Size.csv"

print("Path to dataset files:", path)


# In[31]:


# Step 1: Initialize SparkSession
spark = SparkSession.builder \
    .appName("Text Processing Pipeline") \
    .getOrCreate()


# In[32]:


# Load DataFrame and clean null/empty Text rows
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("multiLine", True) \
    .option("escape", '"') \
    .csv(path) \
    .filter(col("Text").isNotNull() & (col("Text") != "")) \
    .withColumn("doc_id", monotonically_increasing_id())
df.cache()  # Cache DataFrame to avoid recomputation
df.show()


# In[33]:


N = df.count()
print("N = ", N)


# # 2. Tokenization

# In[34]:


from nltk.corpus import stopwords
from nltk import download

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, DoubleType
import re



# Download and load stopwords once
download('stopwords')  # Ensures data exists
STOPWORDS = set(stopwords.words('english'))  # Load into memory


# In[35]:


def tokenize(doc):
    if doc is None or not isinstance(doc, str) or doc.strip() == "":
        return []
    punctuation_removed = re.sub(r'[^a-zA-Z0-9\s]', '', doc.lower())
    tokenized = re.split(r'[\s\n]+', punctuation_removed)
    return [w for w in tokenized if w not in STOPWORDS and len(w) > 0]


# In[36]:


tokenize_udf = udf(tokenize, ArrayType(StringType()))


# In[37]:


tokenized_rdd = df.select(df["doc_id"], tokenize_udf(df["Text"]).alias("tokens")) \
    .rdd.map(lambda x: (x[0], x[1]))
tokenized_rdd.cache()  # Cache RDD to improve performance
tokenized_rdd.take(5)


# # 3. TF/IDF Pipeline

# In[38]:


from collections import Counter


# In[39]:


bag_of_words = tokenized_rdd.mapValues(lambda x: (Counter(x), len(x)))


# In[40]:


global_tf = bag_of_words.flatMap(lambda x: [(word, x[0], local_count, local_count / x[1][1]) for word, local_count in x[1][0].items()])
tf_df = global_tf.toDF(["token", "doc_id", "local_count", "tf"])
tf_df.cache()
tf_df.printSchema()


# In[41]:


token_global_df = tf_df.groupBy("token").count()


# In[42]:


idf = udf(lambda x: math.log(N / x), DoubleType())


# In[43]:


idf_df = token_global_df.withColumn("idf", idf(token_global_df["count"])).select("token", "idf")
idf_df.cache()


# In[44]:


tf_idf = udf(lambda tf, idf: tf * idf, DoubleType())


# In[45]:


tf_idf_df = tf_df.join(idf_df, "token")
main_df = tf_idf_df.withColumn("tf_idf", tf_idf(tf_idf_df["tf"], tf_idf_df["idf"])).select("token", "doc_id", "local_count", "tf_idf")
main_df.cache()


# # (For Testing) Materialization Zone

# In[46]:


# For DataFrames
df_to_be_displayed = main_df
df_to_be_displayed.show()


# In[47]:


# For RDD
rdd_to_be_displayed = bag_of_words
rdd_to_be_displayed.take(5)


# # 4. LDA

# In[ ]:


# ==============================================================================
# Consolidated LDA Implementation, Tuning (K), and Coherence Evaluation Code
# VERSION 5: Fixed prep_start_time definition
# ==============================================================================

import numpy as np
import random
import time # Make sure time is imported
import math
from itertools import combinations
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, collect_list, struct, monotonically_increasing_id, udf
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, StructType, StructField, StringType
from pyspark import StorageLevel

# --- NLTK Imports and Downloads (Add near top imports) ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

print("--- Setting up LDA Tuning Environment ---")

# =================================================
# Section 0: Prerequisites Check & Setup (REVISED)
# =================================================

# --- NLTK Downloads ---
print("Checking and downloading NLTK resources if necessary...")
try:
    STOPWORDS = set(stopwords.words('english'))
    print(" - Stopwords found.")
except LookupError:
    print(" - Downloading stopwords...")
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    print(" - Stopwords downloaded.")
try:
    nltk.data.find('corpora/wordnet')
    print(" - WordNet found.")
except LookupError:
    print(" - Downloading WordNet...")
    nltk.download('wordnet', quiet=True)
    print(" - WordNet downloaded.")
print("NLTK resource check complete.")

# --- Essential variable checks ---
# Assumes spark, df, N, tf_df exist from previous cells
required_vars = ['spark', 'df', 'N', 'tf_df']
prerequisites_ok = True
for var_name in required_vars:
    if var_name not in globals():
        print(f"ERROR: Prerequisite '{var_name}' is not defined. Please run previous cells.")
        prerequisites_ok = False
    else:
        print(f"Prerequisite '{var_name}' available.")
if not prerequisites_ok:
    raise NameError("Missing prerequisite variables.")

# --- Define REVISED Tokenization/Lemmatization Function ---
def tokenize_and_lemmatize(doc):
    lemmatizer = WordNetLemmatizer()
    if doc is None: return []
    try:
        text = doc.lower()
        text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
        tokens = re.split(r'\s+', text)
        lemmatized_tokens = [
            lemmatizer.lemmatize(w) for w in tokens
            if w not in STOPWORDS and len(w) > 2 # Min word length > 2
        ]
        return lemmatized_tokens
    except Exception as e:
        # print(f"Error tokenizing/lemmatizing doc: {e}. Doc: {doc[:50]}...") # Reduce verbosity
        return []

tokenize_udf = udf(tokenize_and_lemmatize, ArrayType(StringType()))
print("Tokenization/Lemmatization UDF defined.")

# --- Apply Tokenization/Lemmatization ---
print("Applying tokenization and lemmatization...")
# Start timing *before* the first Spark action related to prep
prep_start_time = time.time()

tokenized_rdd = df.select("doc_id", tokenize_udf(col("Text")).alias("tokens")) \
                  .rdd.map(lambda row: (row.doc_id, row.tokens))
tokenized_rdd.persist(StorageLevel.MEMORY_AND_DISK)
tokenized_count = tokenized_rdd.count() # Action to trigger computation and caching
print(f"Created and cached 'tokenized_rdd' with {tokenized_count} documents.")

# --- Calculate Document Frequencies and Filter Vocabulary ---
print("Calculating document frequencies for vocabulary filtering...")
doc_unique_words_rdd = tokenized_rdd.mapValues(lambda words: list(set(words))).cache()
word_doc_counts_rdd = doc_unique_words_rdd.flatMap(lambda x: [(word, 1) for word in x[1]])
word_doc_freq_rdd = word_doc_counts_rdd.reduceByKey(lambda a, b: a + b).cache()

min_df = 5
max_df_ratio = 0.85
max_doc_count = N * max_df_ratio
print(f"Filtering vocabulary: min_df={min_df}, max_df_ratio={max_df_ratio:.2f} (max_doc_count={int(max_doc_count)})")
filtered_word_doc_freq_rdd = word_doc_freq_rdd.filter(
    lambda word_count: word_count[1] >= min_df and word_count[1] <= max_doc_count
)

# --- Create Filtered Vocabulary and Mappings ---
print("Creating filtered vocabulary maps...")
filtered_vocabulary_rdd = filtered_word_doc_freq_rdd.map(lambda x: x[0]).zipWithIndex()
V = filtered_vocabulary_rdd.count()
print(f"Filtered Vocabulary Size (V): {V}")
if V == 0: raise ValueError("Vocabulary is empty after filtering!")

vocab_map = filtered_vocabulary_rdd.collectAsMap()
vocab_broadcast = spark.sparkContext.broadcast(vocab_map)
index_to_word_map = {v: k for k, v in vocab_map.items()}
index_to_word_broadcast = spark.sparkContext.broadcast(index_to_word_map)
print("Filtered vocabulary maps created and broadcasted.")

# --- Filter Original Tokens and Prepare Final Input RDD ---
print("Filtering tokens and creating final LDA input RDD...")
filtered_tokenized_rdd = tokenized_rdd.mapValues(
    lambda tokens: [token for token in tokens if token in vocab_broadcast.value]
).cache()

def get_filtered_word_id(token):
    return vocab_broadcast.value.get(token, -1)
get_filtered_word_id_udf = udf(get_filtered_word_id, IntegerType())

temp_word_counts_rdd = filtered_tokenized_rdd.flatMap(
    lambda x: [( (x[0], token), 1 ) for token in x[1]]
).reduceByKey(lambda a, b: a + b)

word_counts_df_filtered = temp_word_counts_rdd.map(
    lambda x: Row(doc_id=x[0][0], token=x[0][1], local_count=x[1])
).toDF().withColumn("word_id", get_filtered_word_id_udf(col("token"))) \
       .select("doc_id", "word_id", "local_count") \
       .filter(col("word_id") != -1)
word_counts_df_filtered.persist(StorageLevel.MEMORY_AND_DISK)

doc_word_tokens_rdd = word_counts_df_filtered.rdd.flatMap(
    lambda row: [(row.doc_id, row.word_id)] * row.local_count
)
doc_word_tokens_rdd.persist(StorageLevel.MEMORY_AND_DISK)

num_tokens = doc_word_tokens_rdd.count()
print(f"Created and cached final 'doc_word_tokens_rdd' with {num_tokens} filtered tokens.")

# End timing for prep section
prep_duration = time.time() - prep_start_time
print(f"Data preparation (incl. filtering) took {prep_duration:.2f} seconds.")

# --- Cleanup intermediate RDDs ---
word_doc_freq_rdd.unpersist()
doc_unique_words_rdd.unpersist()
filtered_tokenized_rdd.unpersist() # Optional cleanup
word_counts_df_filtered.unpersist()
# Keep 'tokenized_rdd' (original lemmatized) cached for coherence


# =================================================
# Section 1: Core LDA Gibbs Sampling Functions
# =================================================

def sample_partition(partition, K_local, alpha_local, beta_local, V_local,
                     n_kv_broadcast_local, n_k_broadcast_local, n_d_broadcast_local):
    """Performs Gibbs sampling update for a partition of word tokens."""
    current_n_kv = n_kv_broadcast_local.value
    current_n_k = n_k_broadcast_local.value
    current_n_d = n_d_broadcast_local.value

    local_partition_list = list(partition)
    if not local_partition_list: return iter([])

    local_n_dk = {}
    for doc_id, word_id, topic in local_partition_list:
        key = (doc_id, topic)
        local_n_dk[key] = local_n_dk.get(key, 0) + 1

    results = []
    K_alpha_term = K_local * alpha_local
    V_beta_term = V_local * beta_local
    epsilon = 1e-9

    for doc_id, word_id, old_topic in local_partition_list:
        local_n_dk[(doc_id, old_topic)] -= 1
        nd = current_n_d.get(doc_id, 0) - 1
        nd = max(0, nd)

        probabilities = np.zeros(K_local)
        term1_den = nd + K_alpha_term

        for k in range(K_local):
            ndk = local_n_dk.get((doc_id, k), 0)
            nkv = current_n_kv.get((word_id, k), 0)
            nk = current_n_k.get(k, 0)

            term1 = (ndk + alpha_local) / term1_den if term1_den > 0 else 0
            term2_den = nk + V_beta_term
            term2 = (nkv + beta_local) / term2_den if term2_den > 0 else 0
            probabilities[k] = term1 * term2

        prob_sum = np.sum(probabilities)
        if prob_sum <= epsilon:
            new_topic = random.randint(0, K_local - 1)
        else:
            normalized_probs = probabilities / prob_sum
            if abs(normalized_probs.sum() - 1.0) > 1e-6 :
                 normalized_probs /= normalized_probs.sum()
            try:
                new_topic = np.random.choice(K_local, p=normalized_probs)
            except ValueError as e:
                 new_topic = random.randint(0, K_local - 1)

        local_n_dk[(doc_id, new_topic)] = local_n_dk.get((doc_id, new_topic), 0) + 1
        results.append((doc_id, word_id, new_topic))

    return iter(results)


def run_lda_gibbs(doc_word_tokens_rdd_local, K_local, alpha_local, beta_local, iterations_local, V_local, N_local, spark_context):
    """Runs the complete LDA Gibbs sampling process."""
    print(f"Running LDA Gibbs: K={K_local}, alpha={alpha_local:.3f}, beta={beta_local}, iters={iterations_local}")
    lda_internal_start_time = time.time()

    print("  Initializing random topics...")
    doc_word_topic_rdd = doc_word_tokens_rdd_local.map(
        lambda x: (x[0], x[1], random.randint(0, K_local - 1))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    print("  Calculating initial counts...")
    n_kv_rdd = doc_word_topic_rdd.map(lambda x: ((x[1], x[2]), 1)).reduceByKey(lambda a, b: a + b)
    n_k_rdd = n_kv_rdd.map(lambda x: (x[0][1], x[1])).reduceByKey(lambda a, b: a + b)
    n_d_rdd = doc_word_topic_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

    n_kv_map = n_kv_rdd.collectAsMap()
    n_k_map = n_k_rdd.collectAsMap()
    n_d_map = n_d_rdd.collectAsMap()
    n_kv_broadcast = spark_context.broadcast(n_kv_map)
    n_k_broadcast = spark_context.broadcast(n_k_map)
    n_d_broadcast = spark_context.broadcast(n_d_map)
    print(f"  Initial counts collected and broadcasted (n_kv: {len(n_kv_map)}, n_k: {len(n_k_map)}, n_d: {len(n_d_map)})")

    broadcast_history = [(n_kv_broadcast, n_k_broadcast)]

    print(f"  Starting {iterations_local} Gibbs sampling iterations...")
    loop_start_time = time.time()
    for i in range(iterations_local):
        iter_start_time = time.time()
        current_n_kv_broadcast, current_n_k_broadcast = broadcast_history[-1]

        new_doc_word_topic_rdd = doc_word_topic_rdd.mapPartitions(
            lambda p: sample_partition(p, K_local, alpha_local, beta_local, V_local,
                                       current_n_kv_broadcast,
                                       current_n_k_broadcast,
                                       n_d_broadcast)
        ).persist(StorageLevel.MEMORY_AND_DISK)

        new_n_kv_rdd = new_doc_word_topic_rdd.map(lambda x: ((x[1], x[2]), 1)).reduceByKey(lambda a, b: a + b)
        new_n_k_rdd = new_n_kv_rdd.map(lambda x: (x[0][1], x[1])).reduceByKey(lambda a, b: a + b)

        new_n_kv_map = new_n_kv_rdd.collectAsMap()
        new_n_k_map = new_n_k_rdd.collectAsMap()

        n_kv_broadcast = spark_context.broadcast(new_n_kv_map)
        n_k_broadcast = spark_context.broadcast(new_n_k_map)
        broadcast_history.append((n_kv_broadcast, n_k_broadcast))

        old_rdd_to_unpersist = doc_word_topic_rdd
        doc_word_topic_rdd = new_doc_word_topic_rdd
        old_rdd_to_unpersist.unpersist()

        iter_duration = time.time() - iter_start_time
        if (i + 1) % 10 == 0 or i == 0 or i == iterations_local - 1 :
             print(f"    Iteration {i+1}/{iterations_local} finished in {iter_duration:.2f} seconds. (n_kv size: {len(new_n_kv_map)})")
    loop_duration = time.time() - loop_start_time
    print(f"  Gibbs loop finished in {loop_duration:.2f} seconds.")

    final_n_kv_broadcast, final_n_k_broadcast = broadcast_history[-1]
    print("  Extracting final Phi distribution...")
    final_n_kv = final_n_kv_broadcast.value
    final_n_k = final_n_k_broadcast.value
    phi_dist = {}
    V_beta_term = V_local * beta_local
    for k_idx in range(K_local):
        phi_dist[k_idx] = {}
        topic_total_words = final_n_k.get(k_idx, 0)
        denominator = topic_total_words + V_beta_term
        if denominator == 0: denominator = 1e-9
        for v_idx in range(V_local):
             word_topic_count = final_n_kv.get((v_idx, k_idx), 0)
             phi_dist[k_idx][v_idx] = (word_topic_count + beta_local) / denominator

    print("  Cleaning up final LDA RDD and intermediate broadcasts...")
    doc_word_topic_rdd.unpersist()
    n_d_broadcast.destroy()
    for i in range(len(broadcast_history)): # Clean all history including last one
         kv_b, k_b = broadcast_history[i]
         try:
             kv_b.destroy(blocking=False) # Use non-blocking destroy
             k_b.destroy(blocking=False)
         except Exception as e:
             print(f"    Warn: Error destroying historical broadcast {i}: {e}")

    lda_internal_duration = time.time() - lda_internal_start_time
    print(f"LDA Gibbs run completed in {lda_internal_duration:.2f} seconds.")
    return phi_dist


# =================================================
# Section 2: Topic Coherence Calculation Functions
# =================================================

def get_relevant_word_pairs(words_in_doc, all_top_words_broadcast_local):
    """Generates pairs of unique words from a doc, filtering for top words."""
    unique_words_in_doc = set(words_in_doc)
    relevant_top_words_in_doc = unique_words_in_doc.intersection(all_top_words_broadcast_local.value)
    if len(relevant_top_words_in_doc) < 2: return []
    pairs = []
    for w1, w2 in combinations(sorted(list(relevant_top_words_in_doc)), 2):
         pairs.append(((w1, w2), 1))
    return pairs

def calculate_topic_coherence(phi_dist, K_local, tokenized_rdd_local, N_local, V_local, num_top_words_coherence, spark_context):
    """Calculates average NPMI coherence for given topics."""
    print(f"--- Calculating Coherence for K={K_local} ---")
    coherence_internal_start_time = time.time()

    # 1.1 Get Top N Words per Topic (Strings)
    print(f"  Extracting top {num_top_words_coherence} words...")
    top_words_per_topic_strings = {}
    for k in range(K_local):
        sorted_words = sorted(phi_dist[k].items(), key=lambda item: item[1], reverse=True)
        top_words_indices = [word_id for word_id, prob in sorted_words[:num_top_words_coherence]]
        top_words_per_topic_strings[k] = [index_to_word_broadcast.value.get(idx, f"UNKNOWN_IDX_{idx}") for idx in top_words_indices]

    # 1.2 Calculate Word Document Frequencies (Using original lemmatized tokens)
    print("  Calculating document frequencies...")
    if not tokenized_rdd_local.is_cached:
        tokenized_rdd_local.persist(StorageLevel.MEMORY_AND_DISK)
        print("    INFO: tokenized_rdd_local was not cached, persisting now.")

    doc_unique_words_rdd = tokenized_rdd_local.mapValues(lambda words: list(set(words))).persist(StorageLevel.MEMORY_AND_DISK)
    word_doc_counts_rdd = doc_unique_words_rdd.flatMap(lambda x: [(word, 1) for word in x[1]])
    word_doc_freq_rdd = word_doc_counts_rdd.reduceByKey(lambda a, b: a + b)
    action_start_time = time.time()
    word_doc_freq_map = word_doc_freq_rdd.collectAsMap()
    print(f"    Collect word_doc_freq action took: {time.time() - action_start_time:.2f} seconds")
    word_doc_freq_broadcast = spark_context.broadcast(word_doc_freq_map)

    # 1.3 Calculate Word Pair Co-Document Frequencies (Optimized)
    print("  Calculating pair co-document frequencies...")
    all_top_words_set = set(w for words in top_words_per_topic_strings.values() for w in words)
    all_top_words_broadcast = spark_context.broadcast(all_top_words_set)

    word_pair_counts_rdd = tokenized_rdd_local.flatMap(
        lambda doc_data: get_relevant_word_pairs(doc_data[1], all_top_words_broadcast)
    )
    word_pair_cocount_rdd = word_pair_counts_rdd.reduceByKey(lambda a, b: a + b)
    action_start_time = time.time()
    word_pair_cocount_map = word_pair_cocount_rdd.collectAsMap()
    print(f"    Collect word_pair_cocount action took: {time.time() - action_start_time:.2f} seconds")
    word_pair_cocount_broadcast = spark_context.broadcast(word_pair_cocount_map)
    doc_unique_words_rdd.unpersist()

    # 1.4 Calculate NPMI and Average Coherence per Topic
    print("  Calculating NPMI scores...")
    topic_coherence_scores = {}
    epsilon = 1e-12
    doc_freqs = word_doc_freq_broadcast.value
    pair_cocounts = word_pair_cocount_broadcast.value

    overall_npmi_sum = 0
    total_pairs_calculated = 0

    for k in range(K_local):
        topic_words = top_words_per_topic_strings[k]
        npmi_topic_scores = []
        print(f"\n    Topic {k} Top Words: {topic_words}")

        for i in range(num_top_words_coherence):
            for j in range(i + 1, num_top_words_coherence):
                w1 = topic_words[i]
                w2 = topic_words[j]
                count_w1 = doc_freqs.get(w1, 0)
                count_w2 = doc_freqs.get(w2, 0)
                pair_key = tuple(sorted((w1, w2)))
                count_w1_w2 = pair_cocounts.get(pair_key, 0)

                if count_w1 == 0 or count_w2 == 0 or count_w1_w2 == 0: npmi = 0.0
                else:
                    p_w1 = count_w1 / N_local
                    p_w2 = count_w2 / N_local
                    p_w1_w2 = count_w1_w2 / N_local
                    pmi = math.log2((p_w1_w2 + epsilon) / (p_w1 * p_w2 + epsilon))
                    denominator = -math.log2(p_w1_w2 + epsilon)
                    if abs(denominator) < epsilon: npmi = 0.0
                    else: npmi = pmi / denominator
                    npmi = max(-1.0, min(1.0, npmi))
                npmi_topic_scores.append(npmi)

        if not npmi_topic_scores: topic_coherence_scores[k] = 0.0
        else: topic_coherence_scores[k] = sum(npmi_topic_scores) / len(npmi_topic_scores)
        print(f"    Topic {k} Coherence (Avg. NPMI): {topic_coherence_scores[k]:.4f}")
        overall_npmi_sum += sum(npmi_topic_scores)
        total_pairs_calculated += len(npmi_topic_scores)

    average_coherence = overall_npmi_sum / total_pairs_calculated if total_pairs_calculated > 0 else 0.0
    print(f"\n  Overall Average Coherence for K={K_local}: {average_coherence:.4f}")

    # Cleanup coherence calculation broadcasts
    word_doc_freq_broadcast.destroy(blocking=False)
    word_pair_cocount_broadcast.destroy(blocking=False)
    all_top_words_broadcast.destroy(blocking=False)

    coherence_internal_duration = time.time() - coherence_internal_start_time
    print(f"--- Coherence calculation finished in {coherence_internal_duration:.2f} seconds ---")
    return average_coherence


# =================================================
# Section 3: Hyperparameter Tuning Loop
# =================================================

print("\n--- Starting Hyperparameter Tuning Loop (Varying K) ---")
tuning_overall_start_time = time.time()

# --- Define Tuning Parameters ---
k_values_to_test = [5, 15, 25, 27, 30] # Adjust range as needed

base_alpha_heuristic = 0.1
base_beta = 0.01
base_iterations = 100 # Increase if needed, start with 100

num_top_words_for_coherence = 10

# --- Store results ---
tuning_results = {}
phi_results = {}

# --- Main Tuning Loop ---
for current_K in k_values_to_test:
    print(f"\n===== Processing K = {current_K} =====")
    tuning_run_start_time = time.time()

    # Determine alpha
    if isinstance(base_alpha_heuristic, str) and base_alpha_heuristic.lower() == '50/k': current_alpha = 50.0 / current_K
    elif isinstance(base_alpha_heuristic, str) and base_alpha_heuristic.lower() == '1/k': current_alpha = 1.0 / current_K
    else: current_alpha = float(base_alpha_heuristic)
    current_beta = base_beta
    current_iterations = base_iterations

    # --- Run LDA ---
    # Uses filtered doc_word_tokens_rdd and filtered V
    current_phi = run_lda_gibbs(doc_word_tokens_rdd, current_K, current_alpha, current_beta,
                                current_iterations, V, N, spark.sparkContext)
    phi_results[current_K] = current_phi

    # --- Calculate Coherence ---
    # Uses original lemmatized tokenized_rdd and filtered V
    current_average_coherence = calculate_topic_coherence(current_phi, current_K, tokenized_rdd, N, V,
                                                         num_top_words_for_coherence, spark.sparkContext)
    tuning_results[current_K] = current_average_coherence

    tuning_run_duration = time.time() - tuning_run_start_time
    print(f"===== Finished processing K = {current_K} in {tuning_run_duration:.2f} seconds =====")


# --- Analyze Tuning Results ---
tuning_overall_duration = time.time() - tuning_overall_start_time
print("\n--- Tuning Complete ---")
print(f"Total tuning time: {tuning_overall_duration:.2f} seconds")
print("\nK vs. Average Coherence:")
sorted_k = sorted(tuning_results.keys())
for k_val in sorted_k:
    score = tuning_results.get(k_val, float('nan')) # Handle missing results if any
    print(f"  K={k_val}: {score:.4f}")

if tuning_results:
    # Find best K based on highest coherence score
    best_k = max(tuning_results, key=tuning_results.get)
    best_coherence = tuning_results[best_k]
    print(f"\nBest K found: {best_k} with coherence {best_coherence:.4f}")

    print(f"\n--- Top {num_top_words_for_coherence} Words for Best K={best_k} ---")
    best_phi = phi_results.get(best_k) # Get phi for the best K
    if best_phi:
        for k in range(best_k):
             sorted_words = sorted(best_phi[k].items(), key=lambda item: item[1], reverse=True)
             top_words_indices = [word_id for word_id, prob in sorted_words[:num_top_words_for_coherence]]
             topic_words = [index_to_word_broadcast.value.get(idx, f"UNKNOWN_IDX_{idx}") for idx in top_words_indices]
             print(f"  Topic {k}: {topic_words}")
    else:
        print(f"    Could not retrieve Phi results for K={best_k}")

else:
    print("\nNo tuning results found.")
    best_k = None

# Plot the results
if tuning_results:
    plt.figure(figsize=(10, 6))
    plot_k_values = sorted(tuning_results.keys())
    plot_coherence_values = [tuning_results[k_val] for k_val in plot_k_values]
    plt.plot(plot_k_values, plot_coherence_values, marker='o')

    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Average Topic Coherence (NPMI)")
    alpha_str = f"{base_alpha_heuristic:.2f}" if isinstance(base_alpha_heuristic, float) else base_alpha_heuristic
    plt.title(f"LDA Coherence vs. Number of Topics (K)\n(alpha={alpha_str}, beta={base_beta}, iters={base_iterations})")
    plt.xticks(k_values_to_test)
    plt.grid(True)
    if best_k:
        plt.scatter([best_k], [best_coherence], color='red', s=100, label=f'Best K={best_k} ({best_coherence:.3f})', zorder=5)
        plt.legend()
    plt.savefig(f"coherence_k_plot_medium.png")
    plt.close()
print("--- End of Script ---")


# # 5. implement and run scalability experiments
# ## 5.1. Data size and scalability

# In[ ]:


# ==============================================================================
# Section 4: Scalability Experiment (Time vs. Data Size)
# ==============================================================================
# NOTE: This section assumes the previous consolidated block (Sections 0-3)
# has run successfully and the following are defined:
#   - spark: SparkSession
#   - df: Original DataFrame
#   - N: Total number of documents in original df
#   - best_k: The optimal K found from tuning (e.g., 15)
#   - base_alpha_heuristic, base_beta, base_iterations: Parameters used for tuning
#   - tokenize_and_lemmatize: The preprocessing function
#   - min_df, max_df_ratio: Vocabulary filtering parameters used before
#   - run_lda_gibbs: The Gibbs sampling function
#   - plt: matplotlib.pyplot
# ==============================================================================

print("\n--- Starting Scalability Experiment (Time vs. Data Size) ---")
scalability_results = {} # Stores { data_size: lda_execution_time }
overall_scalability_start_time = time.time()

# --- Define Parameters for Scalability Runs ---
# Use the best K found during tuning
if 'best_k' not in globals() or best_k is None:
    print("WARN: 'best_k' not found from tuning, using K=15 as default for scalability.")
    best_k_final = 15
else:
    best_k_final = best_k
print(f"Using K = {best_k_final} for scalability tests.")

# Use the alpha/beta/iterations from the tuning section
# Determine final alpha
if isinstance(base_alpha_heuristic, str) and base_alpha_heuristic.lower() == '50/k':
    alpha_final = 50.0 / best_k_final
elif isinstance(base_alpha_heuristic, str) and base_alpha_heuristic.lower() == '1/k':
    alpha_final = 1.0 / best_k_final
else: # Assume it's a fixed numeric value
    alpha_final = float(base_alpha_heuristic)
beta_final = base_beta
iterations_final = base_iterations
print(f"Using alpha={alpha_final:.3f}, beta={beta_final}, iterations={iterations_final}")

# --- Define Data Sizes to Test ---
# Make sure N (full size) is included, usually last
data_sizes_to_test = [5000, 10000, 15000, N]
# Filter out sizes larger than N if N is small
data_sizes_to_test = [s for s in data_sizes_to_test if s <= N]
if N not in data_sizes_to_test: # Ensure full dataset is tested
    data_sizes_to_test.append(N)
data_sizes_to_test = sorted(list(set(data_sizes_to_test))) # Unique sorted sizes
print(f"Testing data sizes: {data_sizes_to_test}")


# --- Loop Through Data Sizes ---
for current_size in data_sizes_to_test:
    print(f"\n===== Processing Size = {current_size} =====")
    run_start_time = time.time()
    subset_prep_success = False # Flag to track if prep finished

    try:
        # 1. Create Subset of ORIGINAL df
        print(f"  Creating data subset...")
        if current_size == N:
             df_subset = df # Use full df for the last run
             N_subset = N
        else:
             # Calculate fraction carefully to avoid issues if current_size > N
             fraction = min(1.0, current_size / N)
             df_subset = df.sample(False, fraction, seed=42)
             # It's better to rely on the target size rather than the exact sampled count
             N_subset = current_size # Use the target size for calculations like max_df

        # 2. Re-run Preprocessing for the Subset
        print(f"  Preprocessing subset (Target N = {N_subset})...")
        prep_subset_start_time = time.time()

        # Re-apply tokenization/lemmatization
        # Use the globally defined tokenize_udf
        tokenized_rdd_subset = df_subset.select("doc_id", tokenize_udf(col("Text")).alias("tokens")) \
                                 .rdd.map(lambda row: (row.doc_id, row.tokens))
        tokenized_rdd_subset.persist(StorageLevel.MEMORY_AND_DISK)
        tokenized_rdd_subset.count() # Action

        # Recalculate Doc Frequencies for this subset
        doc_unique_words_rdd_subset = tokenized_rdd_subset.mapValues(lambda words: list(set(words))).cache()
        word_doc_counts_rdd_subset = doc_unique_words_rdd_subset.flatMap(lambda x: [(word, 1) for word in x[1]])
        word_doc_freq_rdd_subset = word_doc_counts_rdd_subset.reduceByKey(lambda a, b: a + b).cache()

        # Filter vocabulary based on this subset's frequencies
        max_doc_count_subset = N_subset * max_df_ratio
        filtered_word_doc_freq_rdd_subset = word_doc_freq_rdd_subset.filter(
            lambda wc: wc[1] >= min_df and wc[1] <= max_doc_count_subset
        )

        # Create new Vocab maps for the subset
        filtered_vocabulary_rdd_subset = filtered_word_doc_freq_rdd_subset.map(lambda x: x[0]).zipWithIndex()
        V_subset = filtered_vocabulary_rdd_subset.count() # Subset vocabulary size
        print(f"    Subset Vocab Size (V_subset): {V_subset}")
        if V_subset == 0:
             print("    WARN: Subset vocabulary empty, skipping LDA run for this size.")
             scalability_results[current_size] = float('nan')
             raise StopIteration("Empty Vocabulary") # Use StopIteration to break out of try

        vocab_map_subset = filtered_vocabulary_rdd_subset.collectAsMap()
        vocab_broadcast_subset = spark.sparkContext.broadcast(vocab_map_subset)

        # Filter tokens based on subset vocabulary
        filtered_tokenized_rdd_subset = tokenized_rdd_subset.mapValues(
            lambda tokens: [token for token in tokens if token in vocab_broadcast_subset.value]
        ).cache()

        # Define UDF specific to this subset's broadcast
        def get_filtered_word_id_subset(token): return vocab_broadcast_subset.value.get(token, -1)
        get_filtered_word_id_udf_subset = udf(get_filtered_word_id_subset, IntegerType())

        # Recalculate word counts based on filtered tokens
        temp_word_counts_rdd_subset = filtered_tokenized_rdd_subset.flatMap(
            lambda x: [( (x[0], token), 1 ) for token in x[1]]
        ).reduceByKey(lambda a, b: a + b)

        word_counts_df_filtered_subset = temp_word_counts_rdd_subset.map(
            lambda x: Row(doc_id=x[0][0], token=x[0][1], local_count=x[1])
        ).toDF().withColumn("word_id", get_filtered_word_id_udf_subset(col("token"))) \
               .select("doc_id", "word_id", "local_count") \
               .filter(col("word_id") != -1)
        word_counts_df_filtered_subset.persist(StorageLevel.MEMORY_AND_DISK)

        # Create final input RDD for LDA for this subset
        doc_word_tokens_rdd_subset = word_counts_df_filtered_subset.rdd.flatMap(
            lambda row: [(row.doc_id, row.word_id)] * row.local_count
        )
        doc_word_tokens_rdd_subset.persist(StorageLevel.MEMORY_AND_DISK)
        num_tokens_subset = doc_word_tokens_rdd_subset.count() # Action

        print(f"    Created subset 'doc_word_tokens_rdd' with {num_tokens_subset} tokens.")
        print(f"  Preprocessing subset took {time.time() - prep_subset_start_time:.2f} secs.")
        subset_prep_success = True # Mark prep as successful

        # 3. Time LDA Run for the Subset
        print(f"  Running LDA for subset (K={best_k_final}, V={V_subset}, N={N_subset})...")
        lda_timing_start = time.time()

        # Call run_lda_gibbs with subset's RDD, V, N and BEST fixed K, alpha, beta, iters
        # NOTE: run_lda_gibbs uses V_local and N_local arguments passed to it
        _ = run_lda_gibbs(doc_word_tokens_rdd_subset, best_k_final, alpha_final, beta_final,
                          iterations_final, V_subset, N_subset, spark.sparkContext)

        lda_time = time.time() - lda_timing_start
        print(f"  LDA run for size {current_size} took: {lda_time:.2f} seconds")
        scalability_results[current_size] = lda_time

    except StopIteration as si: # Catch the empty vocab exception
        print(f"    Skipping K={current_size} due to {si}")
    except Exception as e:
        print(f"ERROR processing size {current_size}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        scalability_results[current_size] = float('nan') # Mark as failed
    finally:
        # 4. Cleanup Subset Resources (always attempt cleanup)
        print("  Cleaning up subset resources...")
        # Use try-except for each unpersist/destroy in case RDDs weren't created due to error
        try: tokenized_rdd_subset.unpersist()
        except NameError: pass
        try: doc_unique_words_rdd_subset.unpersist()
        except NameError: pass
        try: word_doc_freq_rdd_subset.unpersist()
        except NameError: pass
        try: vocab_broadcast_subset.destroy(blocking=False)
        except NameError: pass
        try: filtered_tokenized_rdd_subset.unpersist()
        except NameError: pass
        try: word_counts_df_filtered_subset.unpersist()
        except NameError: pass
        try: doc_word_tokens_rdd_subset.unpersist()
        except NameError: pass
        print("  Subset cleanup attempted.")

        run_duration = time.time() - run_start_time
        print(f"===== Finished processing size = {current_size} in {run_duration:.2f} seconds =====")


# --- Plot Scalability Results ---
print("\n--- Scalability Results (Time vs. Data Size) ---")
if scalability_results:
     # Filter out failed runs (nan) before plotting
     valid_sizes = [s for s in data_sizes_to_test if s in scalability_results and not math.isnan(scalability_results[s])]
     valid_times = [scalability_results[s] for s in valid_sizes]

     print("Size vs. LDA Time:")
     for size, t in zip(valid_sizes, valid_times):
          print(f"  Size={size}: {t:.2f} seconds")

     if valid_sizes: # Check if there are any valid results to plot
         plt.figure(figsize=(10, 6))
         plt.plot(valid_sizes, valid_times, marker='o')
         plt.xlabel("Number of Documents (N)")
         plt.ylabel("LDA Execution Time (seconds)")
         plt.title(f"Scalability: LDA Time vs. Data Size\n(K={best_k_final}, alpha={alpha_final:.2f}, beta={beta_final}, iters={iterations_final})")
         plt.grid(True)
         # Set x-axis ticks to the actual sizes tested
         plt.xticks(valid_sizes)
         # Optionally set y-axis limit if needed
         # plt.ylim(bottom=0)
         plt.savefig(f"scalability_plot_medium.png")
         plt.close()
     else:
         print("No valid scalability results to plot.")
else:
     print("No scalability results recorded.")

overall_scalability_duration = time.time() - overall_scalability_start_time
print(f"\nTotal Scalability Experiment Time: {overall_scalability_duration:.2f} seconds")
print("--- End Scalability Experiment ---")


# # 6. MLlib Baseline Comparison 

# In[ ]:


# ===========================================================
# Section 5: MLlib Baseline Comparison
# ===========================================================
# Assumes the main analysis notebook (tuning) has run and defines:
# - spark: SparkSession
# - V: Filtered Vocabulary size (from Section 0)
# - word_counts_df_filtered: DataFrame with 'doc_id', 'word_id', 'local_count' (from Section 0)
#   (Ensure this DF is still available/cached, or recreate if needed)
# - best_k: The best K found from tuning (Section 3)
# - base_iterations: The number of iterations used in tuning (Section 3)
# - index_to_word_broadcast: Broadcast mapping {index: 'word'} (from Section 0)
# - scalability_results: Dictionary from Time vs Size experiment (Section 4)
# - N: Total original document count (from initial setup)
# ===========================================================
print("\n--- Starting MLlib Baseline Comparison ---")
baseline_start_time = time.time()

# Import necessary MLlib components
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA as MLlibLDA
import pyspark.sql.functions as F
import time # Ensure time is imported if running this cell independently later
import math # Ensure math is imported

# --- Define Parameters for MLlib Run ---
# Define BEST_ITERATIONS based on the value used in the successful tuning run
if 'base_iterations' in globals():
    BEST_ITERATIONS = base_iterations
else:
    print("WARN: 'base_iterations' not found from tuning context, using 100 as default for MLlib maxIter.")
    BEST_ITERATIONS = 100 # Fallback default

# Ensure best_k is defined
if 'best_k' not in globals() or best_k is None:
    print("WARN: 'best_k' not found from tuning, using 15 as default for MLlib k.")
    best_k = 15 # Fallback if needed

# Ensure V (Vocabulary size from filtered prep) is defined
if 'V' not in globals() or V is None:
     raise NameError("ERROR: Filtered vocabulary size 'V' is not defined. Please ensure Section 0 ran.")

# Ensure word_counts_df_filtered is defined
if 'word_counts_df_filtered' not in globals():
     raise NameError("ERROR: 'word_counts_df_filtered' DataFrame is not defined. Please ensure Section 0 ran.")

# Ensure index_to_word_broadcast is defined
if 'index_to_word_broadcast' not in globals():
     raise NameError("ERROR: 'index_to_word_broadcast' is not defined. Please ensure Section 0 ran.")

mllib_lda_time_recorded = float('nan') # Initialize time result

try:
    # 1. Prepare Data for MLlib LDA
    print(f"  Preparing data for MLlib LDA (k={best_k}, V={V})...")
    prep_mllib_start_time = time.time()

    # Ensure the input DataFrame is cached before potentially heavy groupBy/RDD conversion
    if not word_counts_df_filtered.is_cached:
         print("  WARN: word_counts_df_filtered not cached, might be slow. Re-caching.")
         word_counts_df_filtered.persist(StorageLevel.MEMORY_AND_DISK)
         word_counts_df_filtered.count() # Action to force caching

    # Group counts by document and create sparse vectors
    # Using RDD map is often more flexible for vector creation
    mllib_input_df = word_counts_df_filtered \
        .groupBy("doc_id") \
        .agg(collect_list(struct(col("word_id"), col("local_count"))).alias("counts")) \
        .rdd \
        .map(lambda row: (row.doc_id, Vectors.sparse(V, sorted([(int(c.word_id), float(c.local_count)) for c in row.counts])))) \
        .toDF(["doc_id", "features"]) # MLlib expects "features" column

    mllib_input_df.persist(StorageLevel.MEMORY_AND_DISK)
    num_mllib_docs = mllib_input_df.count() # Action to materialize and cache
    print(f"  Created MLlib input DataFrame with {num_mllib_docs} documents.")
    print(f"  MLlib data preparation took {time.time() - prep_mllib_start_time:.2f} seconds.")

    # 2. Run MLlib LDA and Time .fit()
    print(f"  Running MLlib LDA (k={best_k}, maxIter={BEST_ITERATIONS})...")
    # MLlib's EM optimizer is default and often faster than online for batch
    mllib_lda = MLlibLDA(k=best_k, maxIter=BEST_ITERATIONS, optimizer='em', seed=42) # Use seed

    mllib_timing_start = time.time()
    mllib_model = mllib_lda.fit(mllib_input_df)
    mllib_time = time.time() - mllib_timing_start

    print(f"  MLlib LDA .fit() took: {mllib_time:.2f} seconds")
    mllib_lda_time_recorded = mllib_time # Record successful time

    # 3. Optional: Display MLlib Topics
    print(f"\n  --- Top 10 Words for MLlib LDA (K={best_k}) ---")
    try:
        topics = mllib_model.describeTopics(10)
        # Need the index_to_word mapping
        topic_summary = topics.rdd.map(lambda row: (row.topic, [index_to_word_broadcast.value.get(idx, f"UNK_{idx}") for idx in row.termIndices])).collect()
        for topic_idx, words in sorted(topic_summary):
            print(f"    MLlib Topic {topic_idx}: {words}")
    except Exception as desc_e:
        print(f"    WARN: Could not describe MLlib topics: {desc_e}")

except Exception as e:
    print(f"!!!!! ERROR during MLlib Baseline: {e} !!!!!")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup MLlib DataFrame
    try:
         mllib_input_df.unpersist()
         print("  Unpersisted MLlib input DataFrame.")
    except NameError: pass # If it wasn't created due to error

baseline_duration = time.time() - baseline_start_time
print(f"--- MLlib Baseline Comparison finished in {baseline_duration:.2f} seconds ---")

# 4. Display Performance Comparison
# Requires 'scalability_results' and 'N' from the previous cell (Section 4)
if 'scalability_results' in globals() and N in scalability_results and not math.isnan(mllib_lda_time_recorded):
     my_lda_time = scalability_results[N]
     print("\n--- Performance Comparison (Full Dataset, LDA part only) ---")
     print(f"  Your Gibbs LDA (K={best_k}, iters={base_iterations}): {my_lda_time:.2f} seconds")
     print(f"  MLlib EM LDA (K={best_k}, maxIter={BEST_ITERATIONS}):  {mllib_lda_time_recorded:.2f} seconds")
     if mllib_lda_time_recorded > 0:
         ratio = my_lda_time / mllib_lda_time_recorded
         print(f"  Ratio (Your Time / MLlib Time): {ratio:.2f}")
     else:
         print("  Cannot calculate ratio (MLlib time is zero or invalid).")
elif 'scalability_results' not in globals() or N not in scalability_results:
     print("\nCould not perform final time comparison: Scalability results missing.")
elif math.isnan(mllib_lda_time_recorded):
     print("\nCould not perform final time comparison: MLlib baseline run failed.")
else:
     print("\nCould not perform final time comparison due to missing data.")


# In[ ]:




