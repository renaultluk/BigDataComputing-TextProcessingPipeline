#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ===========================================================
# NEW NOTEBOOK: LDA Executor Scaling Test
# ===========================================================
# Purpose: Test LDA runtime scaling with varying Spark cores (local mode)
# Instructions:
# 1. Save this entire code block as a new .ipynb file (e.g., lda_executor_scaling.ipynb).
# 2. Place the data file 'df_file.csv' in the same directory.
# 3. UPDATE the BEST_K, BEST_ALPHA, etc. constants below with your tuning results.
# 4. Adjust CORE_COUNTS_TO_TEST based on your machine's cores.
# 5. Run the entire notebook. It will stop/start Spark multiple times.
# ===========================================================

import time
import os
import sys # Import sys for flushing output
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, StructType, StructField, StringType
from pyspark import StorageLevel
import numpy as np
import random
import math
from itertools import combinations
import matplotlib.pyplot as plt
from pyspark.sql import Row
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# --- Configuration ---
# *** UPDATE THESE with your best parameters found from tuning ***
BEST_K = 19
BEST_ALPHA = 0.1
BEST_BETA = 0.01
BEST_ITERATIONS = 100
# **************************************************************

CORE_COUNTS_TO_TEST = [1, 2, 4, 8] # Adjust based on your machine's cores
DATA_FILE_PATH = "df_Mid_Size.csv" # Path to your data
# Use preprocessing params consistent with tuning run
MIN_DF = 5
MAX_DF_RATIO = 0.85
# Set driver/executor memory (adjust based on your machine's RAM)
SPARK_MEMORY = "32g"

print("--- Configuration ---")
print(f"Testing Core Counts: {CORE_COUNTS_TO_TEST}")
print(f"Using K={BEST_K}, Alpha={BEST_ALPHA}, Beta={BEST_BETA}, Iterations={BEST_ITERATIONS}")
print(f"Using min_df={MIN_DF}, max_df_ratio={MAX_DF_RATIO}")
print(f"Spark Memory Config: {SPARK_MEMORY}")
print("-" * 20)


# --- NLTK Downloads (run once at start) ---
print("Checking NLTK resources...")
try: STOPWORDS = set(stopwords.words('english'))
except LookupError: nltk.download('stopwords', quiet=True); STOPWORDS = set(stopwords.words('english'))
try: nltk.data.find('corpora/wordnet');
except LookupError: nltk.download('wordnet', quiet=True)
print("NLTK resources ready.")


# --- Define Helper Functions ---

def tokenize_and_lemmatize(doc):
    """Tokenizes, removes stopwords, lemmatizes."""
    lemmatizer = WordNetLemmatizer()
    if doc is None or not isinstance(doc, str) or doc.strip() == "":
        return []
    try:
        text = doc.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = re.split(r'\s+', text)
        lemmatized_tokens = [
            lemmatizer.lemmatize(w) for w in tokens
            if w not in STOPWORDS and len(w) > 2
        ]
        return lemmatized_tokens
    except Exception as e:
        return []


def sample_partition(partition, K_local, alpha_local, beta_local, V_local,
                     n_kv_broadcast_local, n_k_broadcast_local, n_d_broadcast_local):
    """Performs Gibbs sampling update for a partition."""
    # NOTE: Ensure numpy and random are available/imported in the execution environment
    import numpy as np
    import random

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
    """Runs LDA Gibbs sampling and returns the execution time."""
    print(f"  Running LDA Gibbs: K={K_local}, alpha={alpha_local:.3f}, beta={beta_local}, iters={iterations_local}")
    lda_internal_start_time = time.time()

    print("    Initializing random topics...")
    doc_word_topic_rdd = doc_word_tokens_rdd_local.map(
        lambda x: (x[0], x[1], random.randint(0, K_local - 1))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    print("    Calculating initial counts...")
    n_kv_rdd = doc_word_topic_rdd.map(lambda x: ((x[1], x[2]), 1)).reduceByKey(lambda a, b: a + b)
    n_k_rdd = n_kv_rdd.map(lambda x: (x[0][1], x[1])).reduceByKey(lambda a, b: a + b)
    n_d_rdd = doc_word_topic_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

    # Collect initial counts (can be large, might need optimization for huge datasets)
    try:
        n_kv_map = n_kv_rdd.collectAsMap()
        n_k_map = n_k_rdd.collectAsMap()
        n_d_map = n_d_rdd.collectAsMap()
    except Exception as e:
        print(f"    ERROR collecting initial maps: {e}")
        # Clean up partially created RDDs and return NaN or raise
        doc_word_topic_rdd.unpersist()
        raise # Re-raise the exception

    n_kv_broadcast = spark_context.broadcast(n_kv_map)
    n_k_broadcast = spark_context.broadcast(n_k_map)
    n_d_broadcast = spark_context.broadcast(n_d_map)
    print(f"    Initial counts collected and broadcasted (n_kv: {len(n_kv_map)}, n_k: {len(n_k_map)}, n_d: {len(n_d_map)})")
    sys.stdout.flush() # Force print output

    broadcast_history = [(n_kv_broadcast, n_k_broadcast)]

    print(f"    Starting {iterations_local} Gibbs sampling iterations...")
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

        # Collect new maps (can fail here too)
        try:
             new_n_kv_map = new_n_kv_rdd.collectAsMap()
             new_n_k_map = new_n_k_rdd.collectAsMap()
        except Exception as e:
             print(f"    ERROR collecting maps in iteration {i+1}: {e}")
             # Cleanup RDDs from this iteration and raise
             new_doc_word_topic_rdd.unpersist()
             doc_word_topic_rdd.unpersist() # Previous iteration's RDD
             # Destroy all broadcasts created so far
             n_d_broadcast.destroy(blocking=False)
             for kv_b, k_b in broadcast_history:
                 try: kv_b.destroy(blocking=False); k_b.destroy(blocking=False)
                 except: pass
             raise

        n_kv_broadcast = spark_context.broadcast(new_n_kv_map)
        n_k_broadcast = spark_context.broadcast(new_n_k_map)
        broadcast_history.append((n_kv_broadcast, n_k_broadcast))

        old_rdd_to_unpersist = doc_word_topic_rdd
        doc_word_topic_rdd = new_doc_word_topic_rdd
        old_rdd_to_unpersist.unpersist()

        iter_duration = time.time() - iter_start_time
        if (i + 1) % 20 == 0 or i == 0 or i == iterations_local - 1 : # Print less often
             print(f"      Iter {i+1}/{iterations_local} ({iter_duration:.2f}s). n_kv size: {len(new_n_kv_map)}")
             sys.stdout.flush() # Force print output

    loop_duration = time.time() - loop_start_time
    print(f"    Gibbs loop finished in {loop_duration:.2f} seconds.")

    # --- Phi calculation (optional for timing, but good practice) ---
    # final_n_kv_broadcast, final_n_k_broadcast = broadcast_history[-1]
    # final_n_kv = final_n_kv_broadcast.value
    # final_n_k = final_n_k_broadcast.value
    # phi_dist = {} # ... calculate phi ...

    # --- Cleanup ---
    print("    Cleaning up LDA RDDs and broadcasts...")
    doc_word_topic_rdd.unpersist()
    n_d_broadcast.destroy(blocking=False)
    for kv_b, k_b in broadcast_history:
         try: kv_b.destroy(blocking=False); k_b.destroy(blocking=False)
         except: pass # Ignore errors during cleanup

    lda_internal_duration = time.time() - lda_internal_start_time
    print(f"  LDA Gibbs run completed in {lda_internal_duration:.2f} seconds.")
    return lda_internal_duration # Return the time taken


# --- Storage for results ---
executor_timing_results = {}

# --- Loop for Executor Scaling ---
for core_count in CORE_COUNTS_TO_TEST:
    print(f"\n===== Testing with {core_count} Core(s) =====")
    test_start_time = time.time()
    spark = None # Ensure spark is reset
    prep_success = False
    lda_success = False

    try:
        # --- Create/Restart Spark Session ---
        # Stop previous session if it exists
        if 'spark' in locals() and spark and spark.sparkContext._jsc is not None:
            print("  Stopping previous Spark session...")
            spark.stop()
            time.sleep(3) # Pause to ensure resources are released

        master_config = f"local[{core_count}]"
        spark = SparkSession.builder \
            .appName(f"LDA Executor Test - {core_count} Cores") \
            .config("spark.master", master_config) \
            .config("spark.driver.memory", SPARK_MEMORY) \
            .config("spark.executor.memory", SPARK_MEMORY) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        print(f"  Spark Session created for {master_config}")
        spark.sparkContext.setLogLevel("WARN") # Reduce log verbosity

        # --- Re-run FULL Preprocessing ---
        print("  Preprocessing full dataset...")
        prep_start_time = time.time()
        df = spark.read.option("header", True).option("inferSchema", True).option("multiLine", True).option("escape", '\"').csv(DATA_FILE_PATH) \
            .filter(col("Text").isNotNull() & (col("Text") != "")) \
            .withColumn("doc_id", monotonically_increasing_id())
        df.cache()  # Cache DataFrame to avoid recomputation
        N_full = df.count()

        tokenize_udf = udf(tokenize_and_lemmatize, ArrayType(StringType()))  # Redefine UDF for new session if needed

        tokenized_rdd = df.select("doc_id", tokenize_udf(col("Text")).alias("tokens")) \
                          .rdd.map(lambda row: (row.doc_id, row.tokens))
        tokenized_rdd.persist(StorageLevel.MEMORY_AND_DISK).count()

        doc_unique_words_rdd = tokenized_rdd.mapValues(lambda words: list(set(words))).cache()
        word_doc_counts_rdd = doc_unique_words_rdd.flatMap(lambda x: [(word, 1) for word in x[1]])
        word_doc_freq_rdd = word_doc_counts_rdd.reduceByKey(lambda a, b: a + b).cache()

        max_doc_count = N_full * MAX_DF_RATIO
        filtered_word_doc_freq_rdd = word_doc_freq_rdd.filter(
            lambda wc: wc[1] >= MIN_DF and wc[1] <= max_doc_count
        )

        filtered_vocabulary_rdd = filtered_word_doc_freq_rdd.map(lambda x: x[0]).zipWithIndex()
        V_full = filtered_vocabulary_rdd.count()
        print(f"    Full Vocab Size (V_full): {V_full}")
        if V_full == 0: raise ValueError("Full dataset vocabulary empty!")

        vocab_map = filtered_vocabulary_rdd.collectAsMap()
        vocab_broadcast = spark.sparkContext.broadcast(vocab_map)

        filtered_tokenized_rdd = tokenized_rdd.mapValues(
            lambda tokens: [token for token in tokens if token in vocab_broadcast.value]
        ).cache()

        def get_filtered_word_id(token): return vocab_broadcast.value.get(token, -1)
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
        doc_word_tokens_rdd.persist(StorageLevel.MEMORY_AND_DISK).count()

        print(f"  Preprocessing took {time.time() - prep_start_time:.2f} seconds.")
        prep_success = True

        # --- Time LDA ---
        print(f"  Running LDA with {core_count} core(s)...")
        lda_time = run_lda_gibbs(doc_word_tokens_rdd, BEST_K, BEST_ALPHA, BEST_BETA,
                                 BEST_ITERATIONS, V_full, N_full, spark.sparkContext)
        executor_timing_results[core_count] = lda_time
        lda_success = True
        print(f"  LDA Run Time for {core_count} core(s): {lda_time:.2f} seconds")

    except Exception as e:
        print(f"!!!!! ERROR processing core count {core_count}: {e} !!!!!")
        import traceback
        traceback.print_exc()
        executor_timing_results[core_count] = float('nan') # Mark as failed
    finally:
        # --- Cleanup for this run ---
        print(f"  Cleaning up resources for {core_count} core(s)...")
        # Use try-except for safety as RDDs might not exist if prep failed
        if prep_success: # Only unpersist/destroy if prep seemed okay
            try: tokenized_rdd.unpersist(); print("    Unpersisted tokenized_rdd")
            except: pass
            try: doc_unique_words_rdd.unpersist(); print("    Unpersisted doc_unique_words_rdd")
            except: pass
            try: word_doc_freq_rdd.unpersist(); print("    Unpersisted word_doc_freq_rdd")
            except: pass
            try: vocab_broadcast.destroy(blocking=False); print("    Destroyed vocab_broadcast")
            except: pass
            try: filtered_tokenized_rdd.unpersist(); print("    Unpersisted filtered_tokenized_rdd")
            except: pass
            try: word_counts_df_filtered.unpersist(); print("    Unpersisted word_counts_df_filtered")
            except: pass
            try: doc_word_tokens_rdd.unpersist(); print("    Unpersisted doc_word_tokens_rdd")
            except: pass
        # Stop spark session for this run
        if spark:
            spark.stop()
            print("    Stopped Spark session.")
            spark = None # Ensure it's seen as stopped
            time.sleep(3) # Pause

        print(f"===== Finished testing {core_count} core(s) in {time.time() - test_start_time:.2f} seconds =====")
        sys.stdout.flush() # Force print output


# --- Analyze and Plot Executor Scaling Results ---
print("\n--- Executor Scaling Results (Time vs. Cores) ---")
if executor_timing_results:
    # Filter out failed runs before calculating speedup/plotting
    valid_cores = sorted([c for c, t in executor_timing_results.items() if not math.isnan(t)])
    valid_times = [executor_timing_results[c] for c in valid_cores]

    print("Cores vs. LDA Time:")
    for cores, t in zip(valid_cores, valid_times):
        print(f"  Cores={cores}: {t:.2f} seconds")

    # Calculate Speedup (relative to 1 core time if available)
    if 1 in valid_cores:
        time_1_core = executor_timing_results[1]
        speedup = [time_1_core / t if t > 0 else 0 for t in valid_times]
        print("\nCores vs. Speedup:")
        for cores, sp in zip(valid_cores, speedup):
            print(f"  Cores={cores}: {sp:.2f}x")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1) # Plot time vs cores
        plt.plot(valid_cores, valid_times, marker='o')
        plt.xlabel("Number of Executor Cores")
        plt.ylabel("LDA Execution Time (seconds)")
        plt.title("LDA Time vs. Cores")
        plt.xticks(valid_cores)
        plt.grid(True)

        plt.subplot(1, 2, 2) # Plot speedup vs cores
        plt.plot(valid_cores, speedup, marker='o', label='Actual Speedup')
        plt.plot(valid_cores, valid_cores, linestyle='--', color='grey', label='Ideal Speedup') # Ideal linear speedup
        plt.xlabel("Number of Executor Cores")
        plt.ylabel("Speedup (T_1 / T_N)")
        plt.title("LDA Speedup vs. Cores")
        plt.xticks(valid_cores)
        plt.grid(True)
        plt.legend()

        plt.suptitle(f"Executor Scaling (K={BEST_K}, N={N_full}, iters={BEST_ITERATIONS})")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plt.savefig(f"Executor_Scaling_medium_K_19.png")
        plt.close()

    elif valid_cores: # If 1 core run failed/missing, just plot time
        print("\nCannot calculate speedup because 1-core run result is missing or invalid.")
        plt.figure(figsize=(7, 6))
        plt.plot(valid_cores, valid_times, marker='o')
        plt.xlabel("Number of Executor Cores")
        plt.ylabel("LDA Execution Time (seconds)")
        plt.title(f"LDA Time vs. Cores (K={BEST_K}, N={N_full})")
        plt.xticks(valid_cores)
        plt.grid(True)
        plt.savefig(f"Executor_Scaling_medium_K_19.png")
        plt.close()
    else:
        print("No valid executor scaling results to plot.")

else:
    print("No executor scaling results recorded.")

print("--- End Executor Scaling Experiment ---")


# In[ ]:




