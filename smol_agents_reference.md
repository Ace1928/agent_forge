# Smolagents Reference Guide

## smolagents.cli

Utility script for running CodeAgent with configurable models and tools.

### CLI Functions

- `parse_arguments()`: Parses CLI args for agent configuration
- `load_model(model_type, model_id, api_base=None, api_key=None)`: Creates model instances
- `main(prompt, tools, model_type, model_id, api_base=None, api_key=None, imports=None)`: Runs CodeAgent

### CLI Usage

```bash
python codeagent_runner.py --prompt "What is the speed of a leopard?" --model_type HfApiModel --model_id "gpt-3.5-turbo"
```

## smolagents.utils

Utility module with error handling and helper functions.

### Error Classes

- `AgentError`: Base exception class
- `AgentParsingError`: For parsing errors
- `AgentExecutionError`: For execution errors
- `AgentMaxStepsError`: When max steps exceeded
- `AgentGenerationError`: For generation errors

### Functions

- `escape_code_brackets(text)`: Escapes brackets in code
- `make_json_serializable(obj)`: Converts to JSON-compatible format
- `parse_json_blob(json_blob)`: Extracts JSON from text
- `parse_code_blobs(text)`: Extracts Python code blocks
- `truncate_content(content, max_length=20000)`: Truncates long strings
- `get_source(obj)`: Gets source code of class/callable
- `get_method_source(method)`: Gets method source code
- `instance_to_source(instance, base_cls=None)`: Converts instance to source

### Helper Utilities

- `ImportFinder`: AST visitor for finding imports
- `is_same_method(method1, method2)`: Compares methods by source
- `is_same_item(item1, item2)`: Compares class items
- `encode_image_base64(image)`: Encodes image to base64
- `make_image_url(base64_image)`: Creates data URL for image
- `make_init_file(folder)`: Creates empty `__init__.py`
- `_is_package_available(package_name)`: Checks if package is installed

## smolagents.tools

Framework for creating and managing agent tools.

### Tool Class

- **Attributes**: `name`, `description`, `inputs`, `output_type`
- **Methods**:
  - `forward(*args, **kwargs)`: Core implementation
  - `setup()`: Optional initialization
  - `to_dict()`: Serializes to dictionary
  - `save(output_dir)`: Exports tool code
  - `push_to_hub(repo_id)`: Publishes to Hub
- **Static Creation Methods**:
  - `from_hub(repo_id)`: Loads from Hub
  - `from_code(tool_code)`: Creates from code
  - `from_space(space_id)`: Converts from Space
  - `from_gradio/langchain(tool)`: Converts tools

### Tool Creation Examples

```python
@tool
def search_web(query: str) -> List[Dict]:
  """Search the web for results."""
  return results
```

### Tool Collections

- `ToolCollection`: Manages related tools
  - `from_hub(collection_slug)`: Loads from Hub
  - `from_mcp(server_parameters)`: Loads from MCP

### Pipeline Tools

- `PipelineTool`: For Transformer models
  - Handles model loading, pre/post-processing

### Utilities

- `launch_gradio_demo(tool)`: Creates UI for tools
- `load_tool(repo_id)`: Loads Hub tools
- `get_tools_definition_code(tools)`: Exports tool code

## smolagents.remote_executor

Framework for isolated Python code execution.

### Classes

- **RemotePythonExecutor (ABC)**

  - `run_code_raise_errors(code, return_final_answer)`: Executes code
  - `send_tools(tools)`: Sends tool definitions
  - `send_variables(variables)`: Transmits variables
  - `install_packages(additional_imports)`: Installs dependencies

- **E2BExecutor**

  - Cloud-based execution via E2B service
  - Requires `e2b_code_interpreter` package

- **DockerExecutor**
  - Container-based execution via Docker
  - Requires Docker daemon and packages

### CLI Example

```python
executor = E2BExecutor(additional_imports=["numpy"], logger=logger)
result, logs, is_final = executor("import numpy as np\nnp.random.rand(3,3)")
```

## smolagents.monitoring

Monitoring system for tracking agent performance.

### Monitoring Components

- **Monitor**: Tracks metrics (duration, tokens)
- **LogLevel**: `OFF(-1)`, `ERROR(0)`, `INFO(1)`, `DEBUG(2)`
- **AgentLogger**: Rich console output
  - `log()`, `log_error()`, `log_markdown()`, `log_code()`
  - `log_rule()`, `log_task()`, `log_messages()`
  - `visualize_agent_tree()`: Shows agent hierarchy

## smolagents.models

Interface for multiple LLM backends.

### Message Handling

- **ChatMessage**: Container with `role`, `content`, `tool_calls`
- **MessageRole**: Standard roles enum
- **ChatMessageToolCall**: Structure for tool invocations

### Model Classes

- **Base Model**: Common interface for all models
- **Local Models**:

  - `TransformersModel`: Local HF Transformers
  - `VLLMModel`: Optimized inference
  - `MLXModel`: Apple Silicon acceleration

- **API Models**:
  - `HfApiModel`: Hugging Face Inference API
  - `OpenAIServerModel`: OpenAI-compatible
  - `AzureOpenAIServerModel`: Azure-specific
  - `LiteLLMModel`: Universal adapter

### Usage

```python
model = TransformersModel("Qwen/Qwen2.5-Coder-32B-Instruct")
response = model(
  messages=[{"role": "user", "content": "Write a poem"}],
  temperature=0.7
)
```

### Tool Integration

```python
response = model(
  messages=[{"role": "user", "content": "What is 42*73?"}],
  tools_to_call_from=[calculator]
)
```

## smolagents.agent_types

Type system for agent I/O handling.

### Core Classes

- **AgentType**: Base class for all types

  - Common interface for agent inputs/outputs
  - Provides `to_raw()` and `to_string()` methods

- **AgentText**: Extended string type

  - String-compatible agent output
  - Proper string serialization

- **AgentImage**: Extended image type

  - Supports PIL, paths, bytes, tensors, arrays
  - Automatic notebook display
  - Format-preserving serialization

- **AgentAudio**: Audio type
  - Supports paths, tensors, (samplerate, data) tuples
  - Notebook playback integration
  - Sample rate preservation

### Helper Functions

- `handle_agent_input_types`: Converts to raw formats
- `handle_agent_output_types`: Converts to agent types
- `_AGENT_TYPE_MAPPING`: Registry of type converters

## smolagents.agent

Framework for LLM-driven problem-solving agents.

### Agent Classes

- **MultiStepAgent**: Base ReAct agent

  - Manages conversation, tools, planning
  - Supports serialization and streaming
  - Configurable steps, verbosity, grammar

- **ToolCallingAgent**: Specialized for JSON tool calls

  - Leverages native LLM tool calling
  - Manages tool execution and parameters

- **CodeAgent**: Python code generation and execution
  - Local/remote execution environments
  - Import authorization and security

### Execution Flow

1. Receive task
2. Optional planning phase
3. ReAct loop (reason → act → observe)
4. Generate final answer

```python
agent = CodeAgent(model=model, tools=[calculator, web_search])
result = agent.run("What is the square root of the distance between New York and LA?")
```

### Memory & Planning

- Complete history tracking
- Structured recall for context
- Optional periodic planning
- Plan revision based on results

### Serialization

- `save(output_dir)`: Exports agent
- `from_folder(folder)`: Loads from directory
- `from_hub(repo_id)`: Loads from Hub
- `push_to_hub(repo_id)`: Publishes to Hub

### Prompt Templates

- Customizable templates for different phases
- Jinja2 syntax with variable interpolation

## smolagents.default_tools

Ready-to-use tools for common operations.

### Tools

- **PythonInterpreterTool**: Evaluates Python code
- **FinalAnswerTool**: Marks final solution
- **UserInputTool**: Collects user input
- **DuckDuckGoSearchTool**: Web search via DuckDuckGo
- **GoogleSearchTool**: Web search via Google APIs
- **VisitWebpageTool**: Extracts webpage content
- **SpeechToTextTool**: Transcribes audio

### Default Tools Usage

```python
agent = CodeAgent(
  model=model,
  tools=[
    PythonInterpreterTool(authorized_imports=["numpy"]),
    DuckDuckGoSearchTool(max_results=5)
  ]
)
```

## smolagents.gradio_ui

Web UI for agent interaction via Gradio.

### Components

- **GradioUI**: Creates web interface for any agent

  - `launch(share=True)`: Starts web server
  - `interact_with_agent()`: Processes input
  - `upload_file()`: Handles file uploads

- **stream_to_gradio**: Streams execution to UI
  - Shows reasoning, tool usage, errors, stats

### Gradio UI Usage

```python
agent = CodeAgent(model=model)
ui = GradioUI(agent, file_upload_folder="./uploads")
ui.launch(share=True)
```

### Features

- Real-time execution streaming
- File upload support
- Hierarchical chat display
- Proper formatting for code/media

## smolagents.local_python_executor

Secure Python execution environment.

### LocalPythonExecutor

- Restricted imports and operations
- AST-based execution for safety
- Memory isolation from host

### Safety Features

- Import restrictions
- Operation limits
- Dunder method blocking
- Tool protection
- Error augmentation

### Local Python Executor Usage

```python
executor = LocalPythonExecutor(
  additional_authorized_imports=["numpy"],
  max_print_outputs_length=10000
)
executor.send_tools({"web_search": web_search_tool})
executor.send_variables({"data": [1, 2, 3, 4]})
result, logs, is_final = executor("import numpy as np\nprint(np.mean(data))")
```

## smolagents.memory

Memory system for agent interactions.

### Memory Classes

- **Message**: TypedDict for chat messages
- **ToolCall**: Encapsulates function calls
- **MemoryStep**: Base class for all step types
- **ActionStep**: Records agent actions
- **PlanningStep**: Records planning activities
- **TaskStep**: Stores task descriptions
- **SystemPromptStep**: Stores system prompts
- **AgentMemory**: Main memory manager
  - `reset()`: Clears memory
  - `get_succinct_steps()`: Compact history
  - `get_full_steps()`: Complete history
  - `replay()`: Debug logging
