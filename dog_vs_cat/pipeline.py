import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from dog_vs_cat.config.core import config
from dog_vs_cat.processing.features import embarkImputer
from dog_vs_cat.processing.features import Mapper
from dog_vs_cat.processing.features import age_col_tfr

titanic_pipe=Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.base_config.embarked_var)
     ),
     ##==========Mapper======##
     ("map_sex", Mapper(config.base_config.gender_var, config.base_config.gender_mappings)
      ),
     ("map_embarked", Mapper(config.base_config.embarked_var, config.base_config.embarked_mappings )
     ),
     ("map_title", Mapper(config.base_config.title_var, config.base_config.title_mappings)
     ),
     # Transformation of age column
     ("age_transform", age_col_tfr(config.base_config.age_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.base_config.n_estimators, max_depth=config.base_config.max_depth,
                                      random_state=config.base_config.random_state))
          
     ])

# from sklearn.pipeline import Pipeline
# from scikeras.wrappers import KerasClassifier
# from sklearn.preprocessing import StandardScaler
# from dog_vs_cat.model import build_mobilenet_model
# from dog_vs_cat.processing.data_manager import load_dataset

# def create_pipeline():
#     # Load dataset
#     train_generator, validation_generator = load_dataset()

#     # Wrap the model
#     model = KerasClassifier(model=build_mobilenet_model, input_shape=(150, 150, 3), num_classes=2, epochs=30, batch_size=32)

#     # Create the pipeline
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Placeholder if scaling is needed
#         ('model', model)
#     ])

#     return pipeline, train_generator, validation_generator
