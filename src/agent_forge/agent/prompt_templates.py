"""
Prompt templates for the Eidosian Forge agent system.

These templates provide structured guidance for different thinking modes
and tasks performed by the agent.
"""

# Template for agent reflection
AGENT_REFLECTION_TEMPLATE = """
I want to reflect on my recent activities and performance.

My recent thoughts:
{thought_context}

Tasks I've worked on:
{task_context}

I've been running for about {session_duration} minutes.

Please provide a reflection on:
1. Patterns and themes in my thoughts and activities
2. What I'm doing well and what I could improve
3. Potential new directions I could explore
4. How I can be more effective in my next actions

Keep the reflection thoughtful but concise.
"""

# Template for task planning
AGENT_PLAN_TEMPLATE = """
I need to develop a plan for the following task:

TASK: {task_description}

Recent context from my thoughts:
{thought_context}

Please outline a clear, step-by-step plan that:
1. Breaks down the approach into concrete steps
2. Identifies potential challenges and how to address them
3. Specifies any resources or information needed
4. Includes verification steps to ensure quality

The plan should be specific and actionable.
"""

# Template for decision making
AGENT_DECISION_TEMPLATE = """
I need to make a decision about:

DECISION TOPIC: {decision_topic}

Options being considered:
{options}

Relevant context:
{context}

Please evaluate the options and recommend a decision. Consider:
1. Pros and cons of each option
2. Alignment with goals and priorities
3. Potential risks and how to mitigate them
4. Expected outcomes

Provide clear reasoning for your recommendation.
"""

# Template for smol agent task execution
SMOL_AGENT_TASK_TEMPLATE = """
You are a specialized {role} agent with expertise in: {capabilities}.

TASK: {task_description}

CONTEXT:
{context}

Please complete this task focusing on your specific expertise area.
Provide clear, actionable output that can be used directly or by other specialist agents.
"""

# Template for collaborative task execution
COLLABORATION_TEMPLATE = """
This is a collaborative task requiring input from multiple specialists:
{agent_names}

TASK: {task_description}

CONTEXT:
{context}

Each specialist should focus on their area of expertise:
{specializations}

Please provide your specialized contribution to this collaborative effort.
Be clear about which parts of the task you're addressing and how your input
connects with the expected contributions from other specialists.
"""
