# test_import.py
print("Testing imports...")

try:
    from data_preprocessing import DataPreprocessor
    print("✅ DataPreprocessor imported")
except Exception as e:
    print(f"❌ DataPreprocessor: {e}")

try:
    from content_based import ContentBasedRecommender
    print("✅ ContentBasedRecommender imported")
except Exception as e:
    print(f"❌ ContentBasedRecommender: {e}")

try:
    from collaborative_filtering import CollaborativeFilteringRecommender
    print("✅ CollaborativeFilteringRecommender imported")
except Exception as e:
    print(f"❌ CollaborativeFilteringRecommender: {e}")

try:
    from hybrid_model import HybridRecommender
    print("✅ HybridRecommender imported")
except Exception as e:
    print(f"❌ HybridRecommender: {e}")

try:
    from evaluation import ModelEvaluator
    print("✅ ModelEvaluator imported")
except Exception as e:
    print(f"❌ ModelEvaluator: {e}")

print("\nAll imports tested!")