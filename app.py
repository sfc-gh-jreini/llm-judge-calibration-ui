import streamlit as st
import time
from dotenv import load_dotenv
from typing import Tuple
from trulens.providers.openai import OpenAI as OpenAIProvider
from trulens.feedback import GroundTruthAggregator
from trulens.core import TruSession
from trulens.benchmark.benchmark_frameworks.dataset.beir_loader import (
    TruBEIRDataLoader,
)
from trulens.benchmark.benchmark_frameworks.tru_benchmark_experiment import (
    BenchmarkParams, TruBenchmarkExperiment, create_benchmark_experiment_app,
)

load_dotenv()

session = TruSession()

if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1

def set_form_step(action, step=None):
    if action == 'Next':
        st.session_state['current_step'] = st.session_state['current_step'] + 1
    if action == 'Back':
        st.session_state['current_step'] = st.session_state['current_step'] - 1
    if action == 'Jump':
        st.session_state['current_step'] = step

def wizard_form_header():

    step_options = ['Load Data', 'Configure Feedback', 'Run Experiment']
    current_step = st.radio('Calibration Steps:', step_options, index=st.session_state['current_step'] - 1, horizontal = True)
    
    if current_step == 'Load Data':
        set_form_step('Jump', 1)
    elif current_step == 'Configure Feedback':
        set_form_step('Jump', 2)
    elif current_step == 'Run Experiment':
        set_form_step('Jump', 3)

def wizard_form_body():
    ###### Step 1: Load and persist ground truth experiment data ######
    if st.session_state['current_step'] == 1:
        st.write("Step 1: Load Ground Truth Data")
        dataset_name = st.selectbox('Select Dataset', ['scifact', 'fever', 'hotpotqa'])
        st.session_state['dataset_name'] = dataset_name
        sample_size = st.number_input('Number of samples', min_value=1, value=15)
        if st.button('Load Dataset'):
            with st.spinner('Loading dataset...'):
                beir_data_loader = TruBEIRDataLoader(data_folder="./", dataset_name=dataset_name)
                gt_df = beir_data_loader.load_dataset_to_df(download=True)
                gt_df_sample = gt_df.sample(n=sample_size)
                session.add_ground_truth_to_dataset(
                    dataset_name="beir_scifact",
                    ground_truth_df=gt_df_sample,
                    dataset_metadata={"domain": "Information Retrieval"},
                )
                st.session_state['dataset'] = gt_df_sample
                st.success(f'Successfully loaded {len(gt_df_sample)} samples from {dataset_name} dataset.')
        
        if 'dataset' in st.session_state:
            st.write(st.session_state['dataset'].head())

    ###### Step 2: Configure feedback functions for the experiment ######
    if st.session_state['current_step'] == 2:
        st.write("Step 2: Configure Feedback Functions")
        
        st.session_state['use_feedback'] = st.radio('Select feedback functions:', ['Context Relevance'], index = None)
        use_models = st.multiselect('Select models:', ['gpt-4o', 'gpt-4o-mini','gpt-4'])
        
        if st.button('Configure Feedback'):
            with st.spinner('Configuring feedback function...'):
                feedback_functions = []
                for model in use_models:
                    provider = OpenAIProvider(model_engine=model)
                    
                    def feedback_function(
                        input, output, benchmark_params
                    ) -> Tuple[float, float]:
                        return provider.context_relevance(
                            question=input,
                            context=output,
                            temperature=benchmark_params["temperature"],
                        )

                    feedback_function_dict = {'model': model, 'feedback_function': feedback_function}
                    
                    feedback_functions.append(feedback_function_dict)
                
                st.session_state['feedback_functions'] = feedback_functions
                
                st.success('Feedback functions configured successfully.')

    ###### Step 3: Run experiment and show leaderboard results ######
    if st.session_state['current_step'] == 3:
        st.write("Step 3: Run Calibration Experiments")

        true_labels = []
        for chunks in st.session_state['dataset'].expected_chunks:
            for chunk in chunks:
                true_labels.append(chunk["expected_score"])

        agg_funcs = []
        selected_metrics = st.multiselect('Select metrics:', ['ndcg_at_k','precision_at_k','recall_at_k','ir_hit_rate','mrr','auc','kendall_tau','spearman_correlation','brier_score','ece','mae'])
        selected_k = st.number_input('Choose k value', min_value=1, value=10)
        for metric in selected_metrics:
            if metric == 'ndcg_at_k':
                agg_func = GroundTruthAggregator(true_labels=true_labels, k=selected_k).ndcg_at_k
            elif metric == 'precision_at_k':
                agg_func = GroundTruthAggregator(true_labels=true_labels, k=selected_k).precision_at_k
            elif metric == 'recall_at_k':
                agg_func = GroundTruthAggregator(true_labels=true_labels, k=selected_k).recall_at_k
            elif metric == 'ir_hit_rate':
                agg_func = GroundTruthAggregator(true_labels=true_labels, k = selected_k).ir_hit_rate
            elif metric == 'mrr':
                agg_func = GroundTruthAggregator(true_labels=true_labels).mrr
            elif metric == 'auc':
                agg_func = GroundTruthAggregator(true_labels=true_labels).auc
            elif metric == 'kendall_tau':
                agg_func = GroundTruthAggregator(true_labels=true_labels).kendall_tau
            elif metric == 'spearman_correlation':
                agg_func = GroundTruthAggregator(true_labels=true_labels).spearman_correlation
            elif metric == 'brier_score':
                agg_func = GroundTruthAggregator(true_labels=true_labels).brier_score
            elif metric == 'ece':
                agg_func = GroundTruthAggregator(true_labels=true_labels).ece
            elif metric == 'mae':
                agg_func = GroundTruthAggregator(true_labels=true_labels).mae
            agg_funcs.append(agg_func)

        
        if st.button('Run Experiment'):
            with st.spinner('Running experiment...'):

                results = []
                for feedback_function in st.session_state['feedback_functions']:
                    benchmark_experiment = TruBenchmarkExperiment(
                        feedback_fn=feedback_function['feedback_function'],
                        agg_funcs=agg_funcs,
                        benchmark_params=BenchmarkParams(temperature=0.5),
                    )
                    
                    # Simulated results
                    tru_benchmark = create_benchmark_experiment_app(
                        app_name=f"{st.session_state['use_feedback']}",
                        app_version=f"{feedback_function['model']}",
                        benchmark_experiment=benchmark_experiment,
                    )

                    with tru_benchmark:
                        tru_benchmark.app(st.session_state['dataset'])
                    time.sleep(5)
                st.session_state['results'] = session.get_leaderboard()
                st.success('Experiment completed successfully.')
        
        if 'results' in st.session_state:
            def display_grouped_bar_chart(df):
                metrics = [col for col in df.columns if col.startswith('metric_')]
                # Nicer metric names
                metrics = [col.replace('metric_', '') for col in metrics]
                metrics = [metric.replace('_at_', '@') for metric in metrics]
                # Nicer metric names
                df.columns = [col.replace('metric_', '') for col in df.columns]
                df.columns = [col.replace('_at_', '@') for col in df.columns]
                df = st.session_state['results']
                # Melt the dataframe to long format
                df_melted = df.reset_index().melt(id_vars=['app_name', 'app_version'], value_vars=metrics, var_name='metric', value_name='score')

                # Create the bar chart using Streamlit's st.bar_chart
                st.bar_chart(df_melted, x='metric', y='score', color='app_version', stack=False, 
                                     x_label=f"Calibration Score on {st.session_state['dataset_name']}", y_label=f"metric",
                                     width=700, height=500, use_container_width=False, horizontal=True)
            
            display_grouped_bar_chart(st.session_state['results'])
    
    form_footer_cols = st.columns([5,1,1])
    
    disable_back_button = True if st.session_state['current_step'] == 1 else False
    disable_next_button = True if st.session_state['current_step'] == 3 else False
    
    form_footer_cols[1].button('Back', on_click=set_form_step, args=['Back'], disabled=disable_back_button)
    form_footer_cols[2].button('Next', on_click=set_form_step, args=['Next'], disabled=disable_next_button)

def refresh_experiment_tracker():
    if st.button('Refresh Experiment Tracker'):
        with st.spinner('Refreshing experiment tracker...'):
            session.reset_database()
            # Code to refresh the experiment tracker goes here
            st.success('Experiment tracker refreshed successfully.')

def render_wizard_view():
    with st.expander('', expanded=True):
        wizard_form_header()
        wizard_form_body()
    refresh_experiment_tracker()

# Main app
st.title('Calibrating LLM-as-Judge 👩‍⚖️')
render_wizard_view()
