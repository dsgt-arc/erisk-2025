from pyspark.sql.functions import col, trim, regexp_replace, lower

SELF_REF_PATTERN = (
    r"\b("
    r"i|me|my|mine|myself|"
    r"i'm|im|"
    r"i’ve|ive|"
    r"i’d|id|"
    r"i’ll|ill|"
    r"i am|i was"
    r")\b"
)


def PreprocessSelfReferentialPosts(df):
    """
    Cleans and filters a PySpark DataFrame of Reddit posts for self-referential content.

    Args:
        df (DataFrame): Spark DataFrame with a 'TEXT' column.

    Returns:
        DataFrame: Cleaned and filtered DataFrame.
    """
    # Step 1: Clean text
    df_clean = df.filter(trim(col("TEXT")) != "")
    df_clean = df_clean.withColumn(
        "TEXT", regexp_replace("TEXT", r"http\S+|www\S+", "")
    )
    df_clean = df_clean.withColumn(
        "TEXT", regexp_replace("TEXT", r"\[.*?\]\(.*?\)", "")
    )
    df_clean = df_clean.withColumn("TEXT", regexp_replace("TEXT", "gt;", ""))

    # Step 2: Regex match for self-referential language
    df_flagged = df_clean.withColumn(
        "is_self_ref", lower(col("TEXT")).rlike(SELF_REF_PATTERN)
    )

    # Step 3: Filter self-referential posts
    df_filtered = df_flagged.filter(col("is_self_ref"))

    return df_filtered
