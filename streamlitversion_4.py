import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import streamlit as st
import boto3
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import plotly.graph_objects as go




s3 = boto3.client('s3')
placeholder_photo_url = 'placeholder.jpg'
bucket_name = 'mma-info-bucket'
file_name = 'ufc_fights_ml.csv'
response = s3.get_object(Bucket = bucket_name, Key = file_name)
ufc_base_url = 'https://ufc.com'

st.set_page_config(page_title="Ufc prediction model", layout='centered')
st.title("UFC Prediction Model")
st.write("This model predicts the winner between two inputted ufc fighters by analyzing their historical stats and using an XGBoost model. Do use this for predicting the upcoming ufc events!")

def display_tale_of_the_tape(column, fighter_name, fighter_stats):
    """Displays a formatted Tale of the Tape in a given Streamlit column."""
    
    # Calculate wins and losses to create a record string
    total_fights = fighter_stats.get('total_fights', 0)
    win_percentage = fighter_stats.get('win_percentage', 0)
    wins = round(total_fights * (win_percentage / 100))
    losses = total_fights - wins
    

    
    tape_html = f"""
    <div style="font-size: 0.9rem; margin-left: 33px">
        <strong>Record:</strong> {int(wins)} - {int(losses)}<br>
        <strong>Age:</strong> {fighter_stats.get('age', 0)} years<br>
        <strong>Height:</strong> {fighter_stats.get('height', 0)} in<br>
        <strong>Reach:</strong> {fighter_stats.get('reach', 0)} in<br>
    </div>
    """
    column.markdown(tape_html, unsafe_allow_html=True)



def create_horizontal_gauge(name1, prob1, name2, prob2):
    import plotly.graph_objects as go

    if prob1 >= prob2:
        left_name, left_prob, left_color = name1, prob1, '#7DDA49'
        right_name, right_prob, right_color = name2, prob2, '#ff4b4b'
    else:
        left_name, left_prob, left_color = name2, prob2, '#ff4b4b'
        right_name, right_prob, right_color = name1, prob1, '#7DDA49'

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=['Win Probability'],
        x=[left_prob * 100],
        name=left_name,
        orientation='h',
        marker=dict(color=left_color, line=dict(color=left_color, width=1))
    ))

    fig.add_trace(go.Bar(
        y=['Win Probability'],
        x=[right_prob * 100],
        name=right_name,
        orientation='h',
        marker=dict(color=right_color, line=dict(color=right_color, width=1))
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=True,
        legend=dict(
            orientation="h",    
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=150,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig





def impute_with_random_forest(df, target_col, feature_cols):
    """
    Imputes missing values in `target_col` using a RandomForestRegressor trained on `feature_cols`.
    """
    # Split data into rows with and without missing target
    df_known = df[df[target_col].notnull()]
    df_unknown = df[df[target_col].isnull()]

    # If there are no missing values, return the original DataFrame
    if df_unknown.empty:
        return df

    # Prepare features and target
    X = df_known[feature_cols]
    y = df_known[target_col]

    # Train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Predict missing values
    X_missing = df_unknown[feature_cols]
    y_pred = rf.predict(X_missing)

    # Fill the missing values
    df.loc[df[target_col].isnull(), target_col] = y_pred

    return df

def create_advanced_features(df):

    df['striker_score_a'] = df['fighter_a_sig_strikes_landed_per_min'] * df['fighter_a_sig_strike_accuracy']
    df['striker_score_b'] = df['fighter_b_sig_strikes_landed_per_min'] * df['fighter_b_sig_strike_accuracy']
    df['grappler_score_a'] = df['fighter_a_takedowns_landed_per_fight'] + df['fighter_a_submission_attempts_per_fight']
    df['grappler_score_b'] = df['fighter_b_takedowns_landed_per_fight'] + df['fighter_b_submission_attempts_per_fight']
    
    df['striker_grappler_diff'] = (df['striker_score_a'] - df['grappler_score_a']) - (df['striker_score_b'] - df['grappler_score_b'])

    df['strike_efficiency_diff'] = (
    df['fighter_a_sig_strikes_landed_per_min'] / np.maximum(df['fighter_a_sig_strikes_absorbed_per_min'], 0.1)
) - (
    df['fighter_b_sig_strikes_landed_per_min'] / np.maximum(df['fighter_b_sig_strikes_absorbed_per_min'], 0.1)
)


    #new feature 1 
    df['net_strike_rate_diff'] = (
        df['fighter_a_sig_strikes_landed_per_min'].fillna(0) - df['fighter_a_sig_strikes_absorbed_per_min'].fillna(0)
    ) - (
        df['fighter_b_sig_strikes_landed_per_min'].fillna(0) - df['fighter_b_sig_strikes_absorbed_per_min'].fillna(0)
    )

    #new feature 2
    striker_score_a = df['fighter_a_sig_strikes_landed_per_min'] * (df['fighter_a_sig_strike_accuracy'] / 100)
    striker_score_b = df['fighter_b_sig_strikes_landed_per_min'] * (df['fighter_b_sig_strike_accuracy'] / 100)
    grappler_score_a = df['fighter_a_takedowns_landed_per_fight'] + df['fighter_a_submission_attempts_per_fight']
    grappler_score_b = df['fighter_b_takedowns_landed_per_fight'] + df['fighter_b_submission_attempts_per_fight']
    df['striker_grappler_diff'] = (striker_score_a - grappler_score_a) - (striker_score_b - grappler_score_b)


    #new feature 3
    finish_score_a = (df['fighter_a_ko_tko_win_rate'] / 100 ) + df['fighter_a_submission_attempts_per_fight']    
    finish_score_b = (df['fighter_b_ko_tko_win_rate'] / 100 ) + df['fighter_b_submission_attempts_per_fight']    
    df['finish_rate_diff'] = finish_score_a - finish_score_b


    #new feaature 4
    epsilon = 1e-6
    fighter_a_takedown_acc = df['fighter_a_takedown_attempts_per_fight'] / (df['fighter_a_takedowns_landed_per_fight'] + epsilon)
    fighter_b_takedown_acc = df['fighter_b_takedown_attempts_per_fight'] / (df['fighter_b_takedowns_landed_per_fight'] + epsilon)
    df['takedown_acc_diff'] = fighter_a_takedown_acc - fighter_b_takedown_acc


    #new feature 5
    df['exp_weighted_winrate_diff'] = (
    df['fighter_a_win_percentage'] * df['fighter_a_total_fights'] -
    df['fighter_b_win_percentage'] * df['fighter_b_total_fights']
)
    
    #new feature 6
    df['momentum_score_diff'] = (
    df['fighter_a_recent_wins'] + df['fighter_a_win_streak'] + df['fighter_a_control_time_per_fight']
) - (
    df['fighter_b_recent_wins'] + df['fighter_b_win_streak'] + df['fighter_b_control_time_per_fight']
)
    


   #new feature 7
    df['striking_momentum_diff'] = (
    df['fighter_a_recent_wins'] +
    df['fighter_a_win_streak'] +
    df['fighter_a_sig_strikes_landed_per_min'] +
    df['fighter_a_sig_strike_accuracy']
) - (
    df['fighter_b_recent_wins'] +
    df['fighter_b_win_streak'] +
    df['fighter_b_sig_strikes_landed_per_min'] +
    df['fighter_b_sig_strike_accuracy']
)
    

    #new feature 8
    df['fighter_a_norm_momentum'] = (df['fighter_a_reach'] * df['fighter_a_recent_wins']) / (df['fighter_a_total_fights'] + 1)
    df['fighter_b_norm_momentum'] = (df['fighter_b_reach'] * df['fighter_b_recent_wins']) / (df['fighter_b_total_fights'] + 1)
    df['reach_momentum_diff'] = (df['fighter_a_reach'] * df['fighter_a_recent_wins']) -  (df['fighter_b_reach'] * df['fighter_b_recent_wins'])
                           
    #new feature 9
    df['hybrid_striker_score'] = (
    0.5 * df['striking_momentum_diff'] + 
    0.3 * df['strike_efficiency_diff'] + 
    0.2 * df['sig_strikes_landed_per_min_diff']
)



    df.drop(columns=['striker_score_a', 'striker_score_b', 'grappler_score_a', 'grappler_score_b'], inplace=True)

    return df


@st.cache_data
def load_data():
    
    df = pd.read_csv(response['Body'])

    df['event_date'] = pd.to_datetime(df['event_date'])
    NUMERICAL_FIGHTER_STATS = [
        'age', 'height', 'reach', 'weight', 'total_fights', 'win_percentage',
        'recent_wins', 'win_streak', 'sig_strikes_landed_per_min',
        'sig_strikes_absorbed_per_min', 'sig_strike_accuracy', 'sig_strike_defense',
        'takedowns_landed_per_fight', 'takedown_attempts_per_fight',
        'takedown_defense', 'submission_attempts_per_fight',
        'ko_tko_win_rate', 'control_time_per_fight'
    ]

    all_numerical_raw_cols = [f"fighter_a_{s}" for s in NUMERICAL_FIGHTER_STATS] + [f"fighter_b_{s}" for s in NUMERICAL_FIGHTER_STATS]



    # Exclude reach-related columns
    columns_to_exclude = ['fighter_a_reach', 'fighter_b_reach', 'reach_diff']
    impute_cols = [col for col in all_numerical_raw_cols if col not in columns_to_exclude]

    # Impute only the selected columns
    imputer = SimpleImputer(strategy='median')
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    features_for_reach = [
    'fighter_a_height', 'fighter_a_weight', 'fighter_a_age', 'fighter_a_total_fights'
]

    # Apply the imputation
    df = impute_with_random_forest(df, target_col='fighter_a_reach', feature_cols=features_for_reach)

    # Repeat for fighter_b_reach (optionally use similar b-side features)
    features_for_b_reach = [
        'fighter_b_height', 'fighter_b_weight', 'fighter_b_age', 'fighter_b_total_fights'
    ]

    df = impute_with_random_forest(df, target_col='fighter_b_reach', feature_cols=features_for_b_reach)


    all_diff_features = []
    for stat in NUMERICAL_FIGHTER_STATS:
        diff_col_name = f'{stat}_diff'
        df[diff_col_name] = df[f'fighter_a_{stat}'] - df[f'fighter_b_{stat}']
        all_diff_features.append(diff_col_name) 


    create_advanced_features(df)
    all_diff_features.append('striker_grappler_diff')


    all_diff_features.append('strike_efficiency_diff')

    all_diff_features.append('net_strike_rate_diff'),

    all_diff_features.append('finish_rate_diff')

    all_diff_features.append('takedown_acc_diff')

    all_diff_features.append('exp_weighted_winrate_diff')

    all_diff_features.append('momentum_score_diff')

    all_diff_features.append('striking_momentum_diff')

    all_diff_features.append('reach_momentum_diff')



        
    return df, NUMERICAL_FIGHTER_STATS, all_numerical_raw_cols, all_diff_features, imputer

df, NUMERICAL_STATS, RAW_COLS, DIFF_FEATURES,  IMPUTER = load_data()
    
model_features = DIFF_FEATURES + ['weight_class']
def train_model(df, diff_features):
    model_features = diff_features  + ['weight_class']
    X = df[model_features]
    y = df['outcome']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['weight_class']),
            ('num', 'passthrough', diff_features)
        ])

    model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        # Static parameters
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,

        # Updated hyperparameters from your search
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        subsample=0.9
    ))
])

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, stratify=y, random_state = 5)

    model_pipeline.fit(X_train, y_train)
    return model_pipeline

MODEL = train_model(df, DIFF_FEATURES)


all_known_fighters = pd.concat([df['fighter_a_name'], df['fighter_b_name']]).unique().tolist()

from thefuzz import fuzz 
from thefuzz import process

def find_closest_fighter_name(input_name):
    result = process.extractOne(input_name, all_known_fighters)
    return result[0] if result else None



def get_fighter_stats(fighter_name, full_df_imputed, imputer_fitted_obj, numerical_stats_list, raw_col_names_for_imputer):
    fighter_data_a = full_df_imputed[full_df_imputed['fighter_a_name'] == fighter_name].sort_values(by='event_date', ascending=False)
    fighter_data_b = full_df_imputed[full_df_imputed['fighter_b_name'] == fighter_name].sort_values(by='event_date', ascending=False)
    
    recent_fight = None
    prefix = '' # Initialize prefix to avoid UnboundLocalError in some paths

    if not fighter_data_a.empty and not fighter_data_b.empty:
        if fighter_data_a.iloc[0]['event_date'] >= fighter_data_b.iloc[0]['event_date']:
            recent_fight = fighter_data_a.iloc[0]
            prefix = 'fighter_a_'
        else:
            recent_fight = fighter_data_b.iloc[0]
            prefix = 'fighter_b_'
    elif not fighter_data_a.empty:
        recent_fight = fighter_data_a.iloc[0]
        prefix = 'fighter_a_'
    elif not fighter_data_b.empty:
        recent_fight = fighter_data_b.iloc[0]
        prefix = 'fighter_b_'
    else:
        return None # Fighter not found

    stats = {}
    for stat in numerical_stats_list:
        full_col_name = prefix + stat
        # Check if the specific stat exists and is not NaN in the fighter's latest record
        if full_col_name in recent_fight.index and pd.notna(recent_fight[full_col_name]):
            stats[stat] = recent_fight[full_col_name]
        else:
            # Fallback: if the stat was truly missing for this fighter (e.g., they only had NaNs for it),
            # use the median value from the overall training data as learned by the imputer.
            try:
                col_index_in_imputer = raw_col_names_for_imputer.index(full_col_name)
                stats[stat] = imputer_fitted_obj.statistics_[col_index_in_imputer]
            except ValueError:
                # Should not happen if `raw_col_names_for_imputer` is correctly formed.
                # If it does, a stat isn't in the imputed list, which is an issue.
                print(f"Warning: Stat '{full_col_name}' not found in imputer's training columns. Defaulting to NaN.")
                stats[stat] = np.nan # Or raise an error
    stats['weight_class'] = recent_fight['weight_class']

    return stats


def predict_fight_winner(name1_input, name2_input, weight_class,
                         weight_recent_wins, weight_control_time,
                         weight_momentum, weight_reach_momentum):    # --- Step 1: Establish Canonical Order ---
    # Determine which fighter will be treated as 'A' and which as 'B' for consistent diff calculation
    # We'll use alphabetical order of their exact names to ensure consistency
    if name1_input.lower() < name2_input.lower():
        fighter_a_name_canonical = name1_input
        fighter_b_name_canonical = name2_input
        original_fighter1_is_A = True # Flag to remember original input order
    else:
        fighter_a_name_canonical = name2_input
        fighter_b_name_canonical = name1_input
        original_fighter1_is_A = False # Flag to remember original input order

    # --- Step 2: Get Stats for Canonical Fighters ---
    # Call get_fighter_stats with the canonical names
    stats_a = get_fighter_stats(fighter_a_name_canonical, df, IMPUTER, NUMERICAL_STATS, RAW_COLS)
    stats_b = get_fighter_stats(fighter_b_name_canonical, df, IMPUTER, NUMERICAL_STATS, RAW_COLS)
    
    if stats_a is None or stats_b is None:
        # This error handling should ideally be done BEFORE this function,
        # after find_closest_fighter_name in the main UI logic.
        return "Fighter not found", np.array([0.5, 0.5]) # Return neutral result if stats not found

    # --- Step 3: Build a temporary DataFrame for feature calculation ---
    # This DataFrame `temp_fight_df` will hold raw stats of canonical A and B.
    # It mimics the structure of a single row in your training data.
    temp_raw_data = {}
    for stat in NUMERICAL_STATS:
        temp_raw_data[f'fighter_a_{stat}'] = [stats_a.get(stat, np.nan)]
        temp_raw_data[f'fighter_b_{stat}'] = [stats_b.get(stat, np.nan)]
    
    temp_fight_df = pd.DataFrame(temp_raw_data)
    
    # Ensure all numerical raw columns are imputed within this temporary df
    # (Although get_fighter_stats should already return imputed values, this is defensive)
    imputer_for_temp_df = SimpleImputer(strategy='median')
    temp_fight_df[RAW_COLS] = imputer_for_temp_df.fit_transform(temp_fight_df[RAW_COLS]) # Use RAW_COLS for fit_transform

    # --- Step 4: Re-calculate all `_diff` features for the temporary DataFrame ---
    # This ensures consistency with how `_diff` features were created during training
    for stat in NUMERICAL_STATS:
        diff_col_name = f'{stat}_diff'
        temp_fight_df[diff_col_name] = temp_fight_df[f'fighter_a_{stat}'] - temp_fight_df[f'fighter_b_{stat}']
    
    # --- Step 5: Create Advanced Features on the temporary DataFrame ---
    # This function needs to be able to operate on a single-row DataFrame.
    # It will add the advanced feature columns to `temp_fight_df`.
    temp_fight_df = create_advanced_features(temp_fight_df)

    # --- Step 6: Add the assumed weight class ---
    temp_fight_df['weight_class'] = weight_class

      # --- Step 7: Select and order features for the model ---
    predict_df = temp_fight_df[model_features]

    # Defensive check: Add any missing features
    for col in [f for f in model_features if f not in predict_df.columns]:
        predict_df[col] = 0.0

    # --- APPLY FEATURE WEIGHTS HERE 👇 ---
    if 'recent_wins_diff' in predict_df.columns:
        predict_df['recent_wins_diff'] *= weight_recent_wins
    if 'control_time_per_fight_diff' in predict_df.columns:
        predict_df['control_time_per_fight_diff'] *= weight_control_time
    if 'momentum_score_diff' in predict_df.columns:
        predict_df['momentum_score_diff'] *= weight_momentum
    if 'reach_momentum_diff' in predict_df.columns:
        predict_df['reach_momentum_diff'] *= weight_reach_momentum


    # --- Step 8: Make the prediction ---
    prediction = MODEL.predict(predict_df)[0]
    prediction_proba = MODEL.predict_proba(predict_df)[0] # Gives [prob_class_0, prob_class_1]


    # --- Step 9: Map Prediction Back to Original Input Order ---
    # The model predicts for fighter_a_name_canonical (outcome=1 for win)
    # We need to present results based on name1_input and name2_input
    
    if prediction == 1: # Canonical 'A' wins
        canonical_winner = fighter_a_name_canonical
        prob_canonical_winner = prediction_proba[1]
        prob_canonical_loser = prediction_proba[0]
    else: # Canonical 'B' wins (canonical 'A' loses)
        canonical_winner = fighter_b_name_canonical
        prob_canonical_winner = prediction_proba[0] # Probability that canonical A loses (canonical B wins)
        prob_canonical_loser = prediction_proba[1]

    # Now, assign probabilities and winner based on the user's input order (name1_input, name2_input)
    if original_fighter1_is_A: # If name1_input was the canonical A
        winner_for_display = canonical_winner
        prob_name1 = prob_canonical_winner if canonical_winner == name1_input else prob_canonical_loser
        prob_name2 = prob_canonical_winner if canonical_winner == name2_input else prob_canonical_loser
    else: # If name1_input was the canonical B
        winner_for_display = canonical_winner
        prob_name1 = prob_canonical_winner if canonical_winner == name1_input else prob_canonical_loser
        prob_name2 = prob_canonical_winner if canonical_winner == name2_input else prob_canonical_loser

    # Return the winner's name (matching the input name) and the probabilities for name1_input and name2_input
    return winner_for_display, np.array([prob_name1, prob_name2])

    
import re
import requests
from bs4 import BeautifulSoup

@st.cache_data(show_spinner="Fetching fighter photo")
def get_fighter_photo(exact_name):
    name_list  = re.sub(r'[^a-z0-9\s-]', '', exact_name.lower()).split()
    final_name = '-'.join(name_list)

    fighter_url = f"https://ufc.com/athlete/{final_name}"
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    }

    try:
        response = requests.get(fighter_url, timeout= 10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', alt=exact_name)

        if img_tag and 'src' in img_tag.attrs:
            img_url = img_tag['src'] 
            return img_url  
        else:
            st.warning(f"Could not find profile image for {exact_name}")
            return placeholder_photo_url

    except Exception as e:
        st.error(f"Failed to fetch ufc.com page for {exact_name}")
        return placeholder_photo_url
        


def validate_weight_c_matchup(fighter1, fighter2, weight_class):
    fighter1_classes = set(df[df['fighter_a_name'] == fighter1]['weight_class'].unique()) | set(df[df['fighter_b_name'] == fighter1]['weight_class'].unique())
    fighter2_classes = set(df[df['fighter_a_name'] == fighter2]['weight_class'].unique()) | set(df[df['fighter_b_name'] == fighter2]['weight_class'].unique())
    warnings = []

    if weight_class not in fighter1_classes:
        warnings.append(f"{fighter1} usually fights in: {', '.join(fighter1_classes)}. Expect heavily inaccurate results")
    if weight_class not in fighter2_classes:
        warnings.append(f"{fighter2} usually fights in: {', '.join(fighter2_classes)}. Expect heavily inaccurate results")

    if warnings:
        return "Warning: " + " | ".join(warnings)
    return ""



# --- User Interaction ---
fighter1 = st.text_input("Enter the name of Fighter 1", placeholder='Ilia Topuria')
fighter2 = st.text_input("Enter the name of Fighter 2", placeholder ='Charles Oliveira')

weight_class = st.selectbox("Select Weight class", sorted(df['weight_class'].dropna().unique()))

st.sidebar.header("⚖️ Feature Weights Control")
weight_recent_wins = st.sidebar.slider("Recent Wins Diff Weight", 0.0, 3.0, 1.0, 0.1)
weight_control_time = st.sidebar.slider("Control Time per Fight Diff Weight", 0.0, 3.0, 1.0, 0.1)
weight_momentum = st.sidebar.slider("Momentum Score Diff Weight", 0.0, 3.0, 1.0, 0.1)
weight_reach_momentum = st.sidebar.slider("Reach Momentum Diff Weight", 0.0, 3.0, 1.0, 0.1)

if st.button("Predict"):
    name1 = find_closest_fighter_name(fighter1) # This returns the exact matched name
    name2 = find_closest_fighter_name(fighter2) # This returns the exact matched name


    if name1 and name2:
        # Call predict_fight_winner with the exact matched names
        winner_name_for_display, probabilities_array_for_display = predict_fight_winner(
            name1, name2, weight_class,
            weight_recent_wins, weight_control_time, weight_momentum, weight_reach_momentum
        )


        # Handle cases where predict_fight_winner might return an error string
        if isinstance(winner_name_for_display, str) and "Fighter not found" in winner_name_for_display:
             st.error(winner_name_for_display)
             st.stop() # Stop further execution if names are not found


        stats1_display = get_fighter_stats(name1, df, IMPUTER, NUMERICAL_STATS, RAW_COLS)
        stats2_display = get_fighter_stats(name2, df, IMPUTER, NUMERICAL_STATS, RAW_COLS)


        msg = validate_weight_c_matchup(name1, name2, weight_class)
        if msg.startswith("Warning"):
            st.warning(msg)

        photo1_url = get_fighter_photo(name1)
        photo2_url = get_fighter_photo(name2)

        col1, col_vs, col2 = st.columns([5.5, 4.5, 6.8])

        with col1:
            st.subheader(f":green[{name1}]")
            sub_col1_img, sub_col1_stats = st.columns([1, 2])
            with sub_col1_img:
                st.image(photo1_url, width=150, caption=name1)
            with sub_col1_stats:
                display_tale_of_the_tape(col1, name1, stats1_display) # Use stats1_display here

        with col_vs:
            st.image('tekkenvs.png', width=150) 

        with col2:
            st.subheader(f":red[{name2}]")
            sub_col2_img, sub_col2_stats = st.columns([1, 2])
            with sub_col2_img:
                st.image(photo2_url, width=150, caption=name2)
            with sub_col2_stats:
                display_tale_of_the_tape(col2, name2, stats2_display) # Use stats2_display here

        st.write("---")
        
        prob_fighter1_wins = probabilities_array_for_display[0]
        prob_fighter2_wins = probabilities_array_for_display[1]

        if winner_name_for_display == name1:
            st.markdown(f"<h3 style='text-align: center; color:#7DDA49;'>Predicted Winner: {winner_name_for_display}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center; color:#ff4b4b;'>Predicted Winner: {winner_name_for_display}</h3>", unsafe_allow_html=True)

        gauge_fig = create_horizontal_gauge(name1, prob_fighter1_wins, name2, prob_fighter2_wins)
        st.plotly_chart(gauge_fig, use_container_width=True)

    else:
        st.error("One of the two names of the fighters was not found. Please try again")


with st.expander("ℹ️ How the Model Works"):
    st.markdown("""
        This predictor uses a powerful machine learning model called **eXtreme Gradient Boosting (XGBoost)**. 
        
        It works by building a series of decision trees, where each new tree intelligently corrects the errors of the ones that came before it. 
        
        By combining the insights from hundreds of these simple models, it creates a highly accurate final prediction.

        By combining 20+ important features in mma like Sig Strike accuracy, takedown defense, KO/TKO rate etc and training the model on a dataset containing of (constantly updating per event) 8000+ scraped fights gives a fruitful test accuracy of over 75%+ on 10 fold cv.
                
        
    """)


st.sidebar.title("Recent Fights Predictor")
st.sidebar.write("Click the button below to predict the outcomes of the 5 most recent fights in the dataset.")

# 2. Create a button that, when clicked, finds and predicts the latest fights
if st.sidebar.button("Predict 5 Most Recent Fights"):
    st.sidebar.header("Latest Fight Predictions")

    # 3. Sort the DataFrame by date and get the top 5 fights
    df['event_date'] = pd.to_datetime(df['event_date'])
    df.drop_duplicates(inplace=True)

    latest_fights = df.sort_values(by='event_date', ascending=False).head(5)

    # 4. Loop through each of the 5 fights and make a prediction
    for index, fight in latest_fights.iterrows():
        f1 = fight["fighter_a_name"]
        f2 = fight["fighter_b_name"]
        wc = fight["weight_class"]

        # Display the matchup
        st.sidebar.subheader(f"{f1} vs. {f2}")

        # Use your existing prediction function
        winner_name, probabilities = predict_fight_winner(f1, f2, wc, weight_recent_wins, weight_control_time, weight_momentum, weight_reach_momentum )
          # Calculate and display the results
        confidence = max(probabilities) * 100
        st.sidebar.markdown(f"**Winner:** :trophy: {winner_name}")
        st.sidebar.write(f"**Confidence:** {confidence:.2f}%")
        st.sidebar.markdown("---") # Add a separator for the next fight