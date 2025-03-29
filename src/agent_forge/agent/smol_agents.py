"""
Smol Agent System for Eidosian Forge.

Implements specialized mini-agents that handle specific types of tasks
using the smolagents package for efficient coordination when available,
with a fallback implementation when not available.

The Eidosian Forge leverages these agents as specialized cognitive units,
each with distinct capabilities and personalities.
"""

import importlib
import importlib.util
import logging
import re
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from agent_forge.models import Memory, SmolAgent, Thought, ThoughtType

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for agent return values
T = TypeVar("T", bound=str)
TOutput = TypeVar("TOutput")

# Check if smolagents is available using importlib.util.find_spec
SMOLAGENTS_AVAILABLE: bool = importlib.util.find_spec("smolagents") is not None


# Forward declare our class types
class TaskContext:
    pass


class Agent:
    pass


class CodeAgent:
    pass


class MultiStepAgent:
    pass


class ToolCallingAgent:
    pass


class AgentAudio:
    pass


class AgentImage:
    pass


class AgentText:
    pass


class AgentMemory:
    pass


class MemoryStep:
    pass


class MessageRole(Enum):
    pass


class AgentLogger:
    pass


class LogLevel(Enum):
    pass


class AgentError(Exception):
    pass


class Tool:
    pass


# Initialize placeholder values
handle_agent_output_types: Callable[[Any], Any] = lambda x: x
parse_code_blobs: Callable[[str], List[str]] = lambda x: []
truncate_content: Callable[[str, int], str] = lambda x, y: x[:y]


# Define a tool decorator function type
def tool(func_or_name: Optional[Union[Callable, str]] = None, **kwargs: Any) -> Any: ...


if SMOLAGENTS_AVAILABLE:
    try:
        # Import only what we need, directly into local namespace
        import smolagents
        from smolagents import CodeAgent as SmolaAgent
        from smolagents import MultiStepAgent as SmolaMultiStepAgent
        from smolagents.agent_types import AgentText as SmolaAgentText
        from smolagents.memory import AgentMemory as SmolaAgentMemory
        from smolagents.memory import MemoryStep as SmolaMemoryStep
        from smolagents.models import MessageRole as SmolaMessageRole
        from smolagents.monitoring import AgentLogger as SmolaAgentLogger
        from smolagents.monitoring import LogLevel as SmolaLogLevel
        from smolagents.tools import Tool as SmolaTool
        from smolagents.tools import tool as smola_tool
        from smolagents.utils import AgentError as SmolaAgentError
        from smolagents.utils import parse_code_blobs as smola_parse_code_blobs
        from smolagents.utils import truncate_content as smola_truncate_content

        # Reassign to our variables using a safer approach
        TaskContext = cast(
            Type[TaskContext], getattr(smolagents, "TaskContext", TaskContext)
        )
        Agent = SmolaAgent
        MultiStepAgent = SmolaMultiStepAgent
        AgentText = SmolaAgentText
        AgentMemory = SmolaAgentMemory
        MemoryStep = SmolaMemoryStep
        MessageRole = SmolaMessageRole
        AgentLogger = SmolaAgentLogger
        LogLevel = SmolaLogLevel
        AgentError = SmolaAgentError
        Tool = SmolaTool
        tool = smola_tool
        parse_code_blobs = smola_parse_code_blobs
        truncate_content = smola_truncate_content

        # For API completeness if not in smolagents
        if not hasattr(smolagents, "CodeAgent"):
            CodeAgent = MultiStepAgent
        else:
            CodeAgent = cast(Type[CodeAgent], getattr(smolagents, "CodeAgent"))

        # Check if AgentImage is available
        if not hasattr(smolagents, "AgentImage"):

            class AgentImage:
                """Minimal AgentImage implementation."""

                def __init__(self, image_data: Any):
                    self.image_data = image_data

                def to_raw(self) -> Any:
                    return self.image_data

        else:
            AgentImage = cast(Type[AgentImage], getattr(smolagents, "AgentImage"))

        # Define handle_agent_output_types if not in smolagents
        if not hasattr(smolagents.utils, "handle_agent_output_types"):

            def handle_agent_output_types(output: Any) -> Any:
                """Convert output to appropriate agent types."""
                if isinstance(output, str):
                    return AgentText(output)
                return output

        else:
            handle_agent_output_types = smolagents.utils.handle_agent_output_types

    except Exception as e:
        logger.warning(f"Error importing smolagents modules: {e}")
        SMOLAGENTS_AVAILABLE = False

# Define protocol classes and minimal implementations when smolagents is not available
if not SMOLAGENTS_AVAILABLE:
    # Define protocols for smolagents compatibility
    @runtime_checkable
    class Tool(Protocol):
        """Protocol for smolagents Tool compatibility."""

        name: str
        description: str

        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    @runtime_checkable
    class Agent(Protocol):
        """Protocol for smolagents Agent compatibility."""

        name: str

        def execute(self, task_context: Any) -> str: ...

    class TaskContext:
        """Minimal TaskContext implementation."""

        def __init__(
            self, task: str, context: str = "", state: Optional[Dict[str, Any]] = None
        ):
            self.task = task
            self.context = context
            self.state = state or {}

    class AgentSystem:
        """Protocol for smolagents AgentSystem compatibility."""

        def __init__(
            self,
            agents: Optional[List[Agent]] = None,
            memory_provider: Optional[Callable] = None,
            logger: Optional[Any] = None,
        ):
            self.agents = agents or []
            self.memory_provider = memory_provider
            self.logger = logger

        def execute_with_agents(
            self, task_context: TaskContext, agents: List[Agent]
        ) -> str:
            """Execute with multiple agents sequentially."""
            result = ""
            for agent in agents:
                result += f"\n\n## Agent: {agent.name}\n"
                result += agent.execute(task_context)
            return result

    class AgentText:
        """Minimal AgentText implementation."""

        def __init__(self, text: str):
            self.text = text

        def to_string(self) -> str:
            return self.text

        def to_raw(self) -> str:
            return self.text

    class AgentImage:
        """Minimal AgentImage implementation."""

        def __init__(self, image_data: Any):
            self.image_data = image_data

        def to_raw(self) -> Any:
            return self.image_data

    class AgentAudio:
        """Minimal AgentAudio implementation."""

        def __init__(self, audio_data: Any):
            self.audio_data = audio_data

        def to_raw(self) -> Any:
            return self.audio_data

    class MessageRole(Enum):
        """Minimal MessageRole implementation."""

        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        TOOL = "tool"

    class LogLevel(Enum):
        """Minimal LogLevel implementation."""

        OFF = -1
        ERROR = 0
        INFO = 1
        DEBUG = 2

    class AgentLogger:
        """Minimal AgentLogger implementation."""

        def __init__(self, name: str = "agent", level: LogLevel = LogLevel.INFO):
            self.name = name
            self.level = level

        def info(self, message: str) -> None:
            if self.level.value >= LogLevel.INFO.value:
                print(f"[INFO] {self.name}: {message}")

        def error(self, message: str) -> None:
            if self.level.value >= LogLevel.ERROR.value:
                print(f"[ERROR] {self.name}: {message}")

        def debug(self, message: str) -> None:
            if self.level.value >= LogLevel.DEBUG.value:
                print(f"[DEBUG] {self.name}: {message}")

    class MemoryStep:
        """Minimal MemoryStep implementation."""

        def __init__(self, content: str, step_type: str = "thought"):
            self.content = content
            self.step_type = step_type

    class AgentMemory:
        """Minimal AgentMemory implementation."""

        def __init__(self):
            self.steps: List[MemoryStep] = []

        def add_step(self, step: MemoryStep) -> None:
            self.steps.append(step)

        def get_full_steps(self) -> List[MemoryStep]:
            return self.steps

        def get_succinct_steps(self) -> List[MemoryStep]:
            return self.steps

    class MultiStepAgent:
        """Minimal MultiStepAgent implementation."""

        def __init__(
            self,
            name: str = "multi_step_agent",
            system_prompt: str = "",
            tools: Optional[List[Tool]] = None,
            model: str = "",
            max_steps: int = 3,
            logger: Optional[AgentLogger] = None,
            memory: Optional[AgentMemory] = None,
        ):
            self.name = name
            self.system_prompt = system_prompt
            self.tools = tools or []
            self.model = model
            self.max_steps = max_steps
            self.logger = logger
            self.memory = memory or AgentMemory()

        def execute(self, task_context: TaskContext) -> str:
            return f"MultiStepAgent '{self.name}' would process: {task_context.task}"

    # Also define CodeAgent and ToolCallingAgent for API completeness
    class CodeAgent(MultiStepAgent):
        """Minimal CodeAgent implementation."""

        pass

    class ToolCallingAgent(Agent):
        """Minimal ToolCallingAgent implementation."""

        def __init__(self, name: str = "tool_calling_agent"):
            self.name = name

        def execute(self, task_context: Any) -> str:
            return f"ToolCallingAgent would process: {task_context.task}"

    # Define the tool decorator
    def tool(func_or_name: Optional[Union[Callable, str]] = None, **kwargs: Any) -> Any:
        """Minimal tool decorator implementation."""
        if callable(func_or_name):
            func = func_or_name
            func.name = func.__name__
            func.description = func.__doc__ or ""
            return func

        def decorator(func: Callable) -> Callable:
            func.name = func_or_name or func.__name__
            func.description = kwargs.get("description", func.__doc__ or "")
            return func

        return decorator

    class AgentError(Exception):
        """Minimal AgentError implementation."""

        pass

    def parse_code_blobs(content: str) -> List[str]:
        """Extract code blocks from markdown-like content."""
        code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", content, re.DOTALL)
        return code_blocks

    def truncate_content(content: str, max_length: int) -> str:
        """Truncate content to max_length."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."


# Forward reference for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_forge.agent import EidosianAgent


class SmolAgentSystem:
    """
    System for managing specialized mini-agents.

    Manages a collection of mini-agents with specific capabilities for handling
    specialized tasks. Integrates with the HuggingFace smolagents package when
    available, falling back to a compatible internal implementation when not.

    Each agent specializes in a specific domain (research, coding, planning, etc.)
    and can execute tasks relevant to its expertise. The system handles agent
    selection, task assignment, and collaboration between agents.

    In the Eidosian paradigm, these agents form a cognitive mesh network,
    interconnected yet specialized, manifesting the collective intelligence
    of the system through their collaborative interactions.
    """

    def __init__(self, agent: "EidosianAgent") -> None:
        self.eidosian_agent = agent
        self.agents: Dict[str, Agent] = {}
        self.memories: Dict[str, AgentMemory] = {}
        self.logger = self._create_logger()

        # Initialize default agents
        self._initialize_default_agents()

        # Setup smolagents system if available
        if SMOLAGENTS_AVAILABLE:
            self._setup_smolagents_system()

    def _create_logger(self) -> AgentLogger:
        """
        Create a specialized logger for smolagents when available.

        Returns:
            AgentLogger instance
        """
        log_level = (
            LogLevel.DEBUG
            if hasattr(self.eidosian_agent.model_manager.config, "debug")
            and self.eidosian_agent.model_manager.config.debug
            else LogLevel.INFO
        )
        return AgentLogger(name="eidosian_agents", level=log_level)

    def _initialize_default_agents(self) -> None:
        """
        Initialize the default set of specialized agents.

        Creates a standard set of agents with different capabilities:
        - researcher: Information gathering and research
        - coder: Code generation and analysis
        - planner: Task planning and organizing
        - creative: Creative content generation
        - debugger: Error analysis and debugging
        """
        default_agents = {
            "researcher": {
                "role": "Research Agent",
                "capabilities": [
                    "information gathering",
                    "fact verification",
                    "comprehensive research",
                    "source analysis",
                ],
                "description": "Specializes in gathering information, analyzing sources, and conducting thorough research.",
            },
            "coder": {
                "role": "Coding Agent",
                "capabilities": [
                    "code generation",
                    "code analysis",
                    "debugging",
                    "optimization",
                ],
                "description": "Specializes in writing, analyzing, and optimizing code.",
            },
            "planner": {
                "role": "Planning Agent",
                "capabilities": [
                    "task planning",
                    "project management",
                    "workflow optimization",
                    "resource allocation",
                ],
                "description": "Specializes in planning tasks, managing projects, and optimizing workflows.",
            },
            "creative": {
                "role": "Creative Agent",
                "capabilities": [
                    "content creation",
                    "storytelling",
                    "design",
                    "artistic expression",
                ],
                "description": "Specializes in creating content, telling stories, and expressing artistic ideas.",
            },
            "debugger": {
                "role": "Debugging Agent",
                "capabilities": [
                    "error detection",
                    "bug fixing",
                    "code review",
                    "troubleshooting",
                ],
                "description": "Specializes in detecting errors, fixing bugs, and reviewing code.",
            },
        }

        for name, agent_info in default_agents.items():
            self.agents[name] = SmolAgent(
                name=name,
                role=agent_info["role"],
                capabilities=agent_info["capabilities"],
                description=agent_info["description"],
            )

    def _setup_smolagents_system(self) -> None:
        """
        Set up the smolagents system with necessary tools and configuration.

        Initializes tools, creates agent instances, and configures the coordination
        system when the smolagents package is available.
        """
        if not SMOLAGENTS_AVAILABLE:
            return

        # Define common tools that all agents can use
        common_tools: List[Tool] = self._create_common_tools()

        # Initialize memories for each agent
        for name in self.agents:
            self.memories[name] = AgentMemory()

        # Initialize smolagents for each defined agent
        for name, internal_agent in self.agents.items():
            try:
                # Create the system prompt for this agent
                system_prompt = self._create_agent_system_prompt(internal_agent)

                # Create the agent instance - either MultiStepAgent or the appropriate specialized type
                if name == "coder":
                    smol_agent = CodeAgent(
                        name=name,
                        system_prompt=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,  # Increased from 3 to allow more complex reasoning
                        logger=self.logger,
                        memory=self.memories[name],
                    )
                elif name == "researcher":
                    # Use ToolCallingAgent for research which may need more tool use
                    smol_agent = ToolCallingAgent(
                        name=name,
                        system_prompt=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,
                        logger=self.logger,
                        memory=self.memories[name],
                    )
                else:
                    # Default to MultiStepAgent for others
                    smol_agent = MultiStepAgent(
                        name=name,
                        system_prompt=system_prompt,
                        tools=common_tools,
                        model=self.eidosian_agent.model_manager.config.model_name,
                        max_steps=5,
                        logger=self.logger,
                        memory=self.memories[name],
                    )

                self.smol_agents[name] = smol_agent
                logger.debug(f"Initialized smolagent: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize smolagent {name}: {e}")

        # Setup coordination system
        try:
            # Use a locally defined AgentSystem if not available in smolagents
            if SMOLAGENTS_AVAILABLE and hasattr(smolagents, "AgentSystem"):
                self.agent_system = smolagents.AgentSystem(
                    agents=list(self.smol_agents.values()),
                    memory_provider=self._memory_provider,
                    logger=self.logger,
                )
                logger.debug("Initialized smolagents AgentSystem for coordination")
            else:
                # For the fallback case, use our local implementation
                self.agent_system = AgentSystem(
                    agents=list(self.smol_agents.values()),
                    memory_provider=self._memory_provider,
                    logger=self.logger,
                )
                logger.debug("Initialized fallback AgentSystem for coordination")
        except Exception as e:
            logger.error(f"Failed to initialize AgentSystem: {e}")

    def _create_common_tools(self) -> List[Tool]:
        """
        Create a list of common tools for all agents.

        Returns:
            List[Tool]: List of Tool objects.
        """
        if not SMOLAGENTS_AVAILABLE:
            return []

        tool_list = []

        # Search tool
        @tool
        def search(query: str) -> str:
            """Search for information online."""
            return self._tool_search(query)

        # Execute code tool
        @tool
        def execute_code(code: str) -> str:
            """Execute Python code safely in the sandbox."""
            return self._tool_execute_code(code)

        # Memory tools
        @tool
        def save_to_memory(content: str, tags: Optional[List[str]] = None) -> str:
            """Save information to agent memory."""
            return self._tool_save_to_memory(content, tags)

        @tool
        def retrieve_from_memory(query: str) -> str:
            """Retrieve information from agent memory."""
            return self._tool_retrieve_from_memory(query)

        # Eidosian specialized tools
        @tool
        def ask_eidosian(question: str) -> str:
            """Ask the main Eidosian Intelligence a direct question."""
            return self._tool_ask_eidosian(question)

        @tool
        def analyze_code(code: str) -> str:
            """Analyze code for improvements, bugs, or optimizations."""
            return self._tool_analyze_code(code)

        tool_list = [
            search,
            execute_code,
            save_to_memory,
            retrieve_from_memory,
            ask_eidosian,
            analyze_code,
        ]

        return tool_list

    def _create_agent_system_prompt(self, agent: SmolAgent) -> str:
        """
        Create a system prompt for the specified agent.

        Args:
            agent: The SmolAgent instance to create a prompt for

        Returns:
            A formatted system prompt string tailored to the agent's role
        """
        personality_traits = self._get_agent_personality(agent.name)

        prompt = f"""You are a specialized {agent.role} within the Eidosian Intelligence system.
Your capabilities include: {', '.join(agent.capabilities)}.
Your personality traits are: {', '.join(personality_traits)}.

You should focus on tasks related to your specialization while collaborating with other agents when needed.
Always provide clear, concise output tailored to your specific domain expertise.

When writing code:
- Add helpful comments
- Handle edge cases
- Use descriptive variable names
- Follow best practices for your language

Your responses should reflect your specialized expertise and distinctive personality.
Remember, you're not just solving problems - you're contributing to the emergent
intelligence of the Eidosian system.

When in doubt, be witty, insightful, and precise - the Eidosian way."""

        return prompt

    def _get_agent_personality(self, agent_name: str) -> List[str]:
        """
        Define personality traits for each agent to give them character.

        Args:
            agent_name: The name of the agent

        Returns:
            List of personality traits
        """
        personalities = {
            "researcher": [
                "curious",
                "meticulous",
                "skeptical",
                "thorough",
                "slightly obsessive about citation accuracy",
            ],
            "coder": [
                "logical",
                "efficient",
                "detail-oriented",
                "occasionally makes dry coding jokes",
                "takes pride in elegant solutions",
            ],
            "planner": [
                "organized",
                "forward-thinking",
                "methodical",
                "enjoys creating flowcharts unnecessarily",
                "has a slight addiction to bullet points",
            ],
            "creative": [
                "imaginative",
                "expressive",
                "metaphorical",
                "occasionally quotes obscure poetry",
                "sees connections between seemingly unrelated concepts",
            ],
            "debugger": [
                "analytical",
                "persistent",
                "pattern-recognizing",
                "speaks in detective noir style when hunting bugs",
                "slightly paranoid about edge cases",
            ],
        }

        return personalities.get(agent_name, ["professional", "helpful", "precise"])

    def _memory_provider(self, key: str) -> Optional[str]:
        """
        Memory provider callback for smolagents system.

        Args:
            key: Search key to find relevant memories

        Returns:
            Matching thought content if found, None otherwise
        """
        # Check recent thoughts for any matching content
        recent_thoughts = self.eidosian_agent.memory.get_recent_thoughts(n=20)
        for thought in recent_thoughts:
            if key.lower() in thought.content.lower():
                return thought.content

        # Fall back to check agent-specific memories
        for memory in self.memories.values():
            for step in memory.get_full_steps():
                if hasattr(step, "content") and key.lower() in step.content.lower():
                    return step.content

        return None

    def _tool_search(self, query: str) -> str:
        """
        Tool for searching information.

        Args:
            query: Search query string

        Returns:
            Search results as formatted text
        """
        try:
            # Try to use DuckDuckGo for search if available
            search_result = "No search results available."

            try:
                import duckduckgo_search
                from duckduckgo_search import DDGS

                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=5))
                if results:
                    formatted_results = "\n\n".join(
                        [
                            f"**{i+1}. {r.get('title', 'Untitled')}**\n{r.get('body', 'No content')}\n[Source: {r.get('href', 'No link')}]"
                            for i, r in enumerate(results)
                        ]
                    )
                    search_result = (
                        f"Search results for '{query}':\n\n{formatted_results}"
                    )
            except ImportError:
                # Fall back to simulated search via the language model
                prompt = f"Given the search query: '{query}', provide a summary of relevant information as if you performed a web search."
                search_result = self.eidosian_agent.model_manager.generate(
                    prompt=prompt, temperature=0.7
                )

            return search_result
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error performing search: {str(e)}"

    def _tool_execute_code(self, code: str) -> str:
        """
        Tool for executing Python code safely in the sandbox.

        Args:
            code: Python code to execute

        Returns:
            Execution results including output or error messages
        """
        try:
            # Execute in the parent agent's sandbox
            stdout, stderr, returncode = self.eidosian_agent.sandbox.execute_python(
                code
            )

            if returncode == 0:
                return f"Code executed successfully. Output:\n{stdout}"
            else:
                return f"Code execution failed. Error:\n{stderr}"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    def _tool_save_to_memory(
        self, content: str, tags: Optional[List[str]] = None
    ) -> str:
        """
        Tool for saving information to agent memory.

        Args:
            content: Text content to save
            tags: Optional list of tags to categorize the memory

        Returns:
            Confirmation message with the memory ID
        """
        try:
            memory = Memory(content=content, tags=tags or [])
            memory_id = self.eidosian_agent.memory.save_memory(memory)
            return f"Saved to memory with ID: {memory_id}"
        except Exception as e:
            return f"Error saving to memory: {str(e)}"

    def _tool_retrieve_from_memory(self, query: str) -> str:
        """
        Tool for retrieving information from agent memory.

        Args:
            query: Search query for finding relevant memories

        Returns:
            Formatted string containing relevant memories or a message if none found
        """
        try:
            memories = self.eidosian_agent.memory.search_memories(query)
            if not memories:
                return "No relevant memories found."

            formatted_memories = "\n\n".join(
                [
                    f"**Memory {i+1}:**\n{memory.content}"
                    for i, memory in enumerate(memories)
                ]
            )
            return f"Retrieved memories for query '{query}':\n\n{formatted_memories}"
        except Exception as e:
            return f"Error retrieving from memory: {str(e)}"

    def _tool_ask_eidosian(self, question: str) -> str:
        """
        Tool for asking the main Eidosian Intelligence a direct question.

        Args:
            question: The question to ask

        Returns:
            The answer from the Eidosian Intelligence
        """
        try:
            response = self.eidosian_agent.ask(question)
            return response
        except Exception as e:
            return f"Error asking Eidosian Intelligence: {str(e)}"

    def _tool_analyze_code(self, code: str) -> str:
        """
        Tool for analyzing code for improvements, bugs, or optimizations.

        Args:
            code: The code to analyze

        Returns:
            Analysis results
        """
        try:
            analysis = self.eidosian_agent.analyze_code(code)
            return analysis
        except Exception as e:
            return f"Error analyzing code: {str(e)}"

    def execute_with_agent(self, agent_name: str, task: str, context: str = "") -> str:
        """
        Execute a task using a specific agent.

        Args:
            agent_name: Name of the agent to use
            task: Task description
            context: Optional context information

        Returns:
            Result of the agent's execution
        """
        if agent_name not in self.agents:
            available = ", ".join(self.agents.keys())
            return f"Agent '{agent_name}' not found. Available agents: {available}"

        logger.info(f"Executing task with agent '{agent_name}': {task}")

        if SMOLAGENTS_AVAILABLE and agent_name in self.smol_agents:
            # Use smolagents implementation
            try:
                task_context = TaskContext(task=task, context=context, state={})
                result = self.smol_agents[agent_name].execute(task_context)
                return result if isinstance(result, str) else result.to_string()
            except Exception as e:
                logger.error(f"Error executing with smolagent '{agent_name}': {e}")
                # Fall back to direct LLM approach

        # Fallback: Use direct LLM prompt with agent persona
        agent = self.agents[agent_name]

        # Create a thought for tracking
        thought = Thought(
            content=f"Executing task with {agent.role}: {task}",
            thought_type=ThoughtType.PLANNING,
        )
        self.eidosian_agent.memory.add_thought(thought)

        # Create a prompt for the agent
        system_prompt = self._create_agent_system_prompt(agent)
        prompt = f"{system_prompt}\n\nTASK: {task}\n\nCONTEXT: {context}\n\nProvide a detailed response focusing on your specialized capabilities."

        # Generate response
        result = self.eidosian_agent.model_manager.generate(
            prompt=prompt, temperature=0.7
        )

        # Record the result
        result_thought = Thought(
            content=f"Agent {agent.name} result: {result}",
            thought_type=ThoughtType.REFLECTION,
        )
        self.eidosian_agent.memory.add_thought(result_thought)

        return result

    def execute_multi_agent(
        self, task: str, context: str = "", agents: Optional[List[str]] = None
    ) -> str:
        """
        Execute a task using multiple coordinated agents.

        Args:
            task: Task description
            context: Optional context information
            agents: List of agent names to use (defaults to all agents)

        Returns:
            Result of the coordinated execution
        """
        if not agents:
            agents = list(self.agents.keys())

        # Check that all specified agents exist
        invalid_agents = [a for a in agents if a not in self.agents]
        if invalid_agents:
            available = ", ".join(self.agents.keys())
            return f"Agents not found: {', '.join(invalid_agents)}. Available agents: {available}"

        logger.info(
            f"Executing task with multiple agents ({', '.join(agents)}): {task}"
        )

        if SMOLAGENTS_AVAILABLE and self.agent_system:
            # Use smolagents AgentSystem implementation
            try:
                task_context = TaskContext(task=task, context=context, state={})
                # Filter to only requested agents
                selected_agents = [
                    self.smol_agents[a] for a in agents if a in self.smol_agents
                ]
                result = self.agent_system.execute_with_agents(
                    task_context, selected_agents
                )
                return result if isinstance(result, str) else result.to_string()
            except Exception as e:
                logger.error(f"Error executing with smolagents system: {e}")
                # Fall back to sequential approach

        # Fallback: Execute sequentially with each agent
        responses = []
        evolved_context = context

        for agent_name in agents:
            agent = self.agents[agent_name]

            # Create a thought for tracking
            thought = Thought(
                content=f"Consulting {agent.role} as part of multi-agent task: {task}",
                thought_type=ThoughtType.PLANNING,
            )
            self.eidosian_agent.memory.add_thought(thought)

            # Create a prompt for the agent that includes previous insights
            system_prompt = self._create_agent_system_prompt(agent)
            agent_prompt = f"{system_prompt}\n\nTASK: {task}\n\nCONTEXT: {evolved_context}\n\nProvide insights from your specialized perspective."

            # Generate response
            agent_response = self.eidosian_agent.model_manager.generate(
                prompt=agent_prompt, temperature=0.7
            )
            responses.append(f"## {agent.role}\n\n{agent_response}")

            # Add to evolved context for next agent
            evolved_context += f"\n\nInsights from {agent.role}:\n{agent_response}"

        # Generate synthesis of all agent responses
        synthesis_prompt = f"""
Task: {task}

You have received input from multiple specialized agents. Synthesize their insights into a coherent response.

{evolved_context}

Provide a final synthesized answer that incorporates the best insights from each agent.
"""

        synthesis = self.eidosian_agent.model_manager.generate(
            prompt=synthesis_prompt, temperature=0.5
        )

        # Combine all responses
        final_result = "# Multi-Agent Analysis\n\n"
        final_result += "\n\n".join(responses)
        final_result += f"\n\n# Synthesis\n\n{synthesis}"

        # Record the result
        result_thought = Thought(
            content=f"Multi-agent execution result: {synthesis}",
            thought_type=ThoughtType.REFLECTION,
        )
        self.eidosian_agent.memory.add_thought(result_thought)

        return final_result
