import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import your main function
from deepkernels.main import main

@patch('deepkernels.main.pd.read_csv')
@patch('deepkernels.main.DataOrchestrator')
@patch('deepkernels.main.StateSpaceKernelProcess')
@patch('deepkernels.main.ParameterIsolate')
@patch('deepkernels.main.LangevinTrainer')
@patch('deepkernels.main.os.path.exists')
def test_main_execution_flow(
    mock_exists, 
    mock_trainer_class, 
    mock_isolate_class, 
    mock_model_class, 
    mock_orchestrator_class, 
    mock_read_csv
):
    """
    Tests that main.py correctly strings together the DataOrchestrator, Model, 
    Optimizers, and Trainer without actually executing the math.
    """
    
    # 1. Bypass the file system checks
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({'dummy': [1, 2, 3]})
    
    # 2. Mock the DataOrchestrator to return fake dataloaders
    mock_orchestrator_instance = mock_orchestrator_class.return_value
    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_orchestrator_instance.run_pipeline.return_value = (mock_train_loader, mock_test_loader)
    
    # 3. Mock ParameterIsolate to return fake optimizers
    mock_isolate_instance = mock_isolate_class.return_value
    mock_adam = MagicMock()
    mock_sgld = MagicMock()
    mock_isolate_instance.seperate_params_and_build_optimisers.return_value = (mock_adam, mock_sgld)
    
    # 4. Mock the Trainer
    mock_trainer_instance = mock_trainer_class.return_value

    # --- EXECUTE MAIN ---
    main()

    # --- ASSERTIONS (The Plumbing Checks) ---
    
    # Did it read both CSVs?
    assert mock_read_csv.call_count == 2
    
    # Did it run the pipeline with the correct sequence length and batch size?
    mock_orchestrator_instance.run_pipeline.assert_called_once()
    _, kwargs = mock_orchestrator_instance.run_pipeline.call_args
    assert kwargs['seq_len'] == 32  # Or 32 if you updated your main.py!
    assert kwargs['batch_size'] == 128
    
    # Did it isolate the parameters?
    mock_isolate_instance.seperate_params_and_build_optimisers.assert_called_once()
    
    # Did it build the trainer with the correct optimizers?
    mock_trainer_class.assert_called_once()
    _, kwargs = mock_trainer_class.call_args
    assert kwargs['adam_optimiser'] == mock_adam
    assert kwargs['sgld_optimiser'] == mock_sgld
    
    # MOST IMPORTANTLY: Did it actually start training with the loaders?
    mock_trainer_instance.fit.assert_called_once_with(mock_train_loader, mock_test_loader)