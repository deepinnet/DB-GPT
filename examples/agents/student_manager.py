from dbgpt_ext.datasource.rdbms.conn_sqlite import SQLiteTempConnector

connector = SQLiteTempConnector.create_temporary_db()
connector.create_temp_tables(
    {
        "students": {
            "columns": {
                "student_id": "INTEGER PRIMARY KEY",
                "student_name": "TEXT",
                "major": "TEXT",
                "year_of_enrollment": "INTEGER",
                "student_age": "INTEGER",
            },
            "data": [
                (1, "Zhang San", "Computer Science", 2020, 20),
                (2, "Li Si", "Computer Science", 2021, 19),
                (3, "Wang Wu", "Physics", 2020, 21),
                (4, "Zhao Liu", "Mathematics", 2021, 19),
                (5, "Zhou Qi", "Computer Science", 2022, 18),
                (6, "Wu Ba", "Physics", 2020, 21),
                (7, "Zheng Jiu", "Mathematics", 2021, 19),
                (8, "Sun Shi", "Computer Science", 2022, 18),
                (9, "Liu Shiyi", "Physics", 2020, 21),
                (10, "Chen Shier", "Mathematics", 2021, 19),
            ],
        },
        "courses": {
            "columns": {
                "course_id": "INTEGER PRIMARY KEY",
                "course_name": "TEXT",
                "credit": "REAL",
            },
            "data": [
                (1, "Introduction to Computer Science", 3),
                (2, "Data Structures", 4),
                (3, "Advanced Physics", 3),
                (4, "Linear Algebra", 4),
                (5, "Calculus", 5),
                (6, "Programming Languages", 4),
                (7, "Quantum Mechanics", 3),
                (8, "Probability Theory", 4),
                (9, "Database Systems", 4),
                (10, "Computer Networks", 4),
            ],
        },
        "scores": {
            "columns": {
                "student_id": "INTEGER",
                "course_id": "INTEGER",
                "score": "INTEGER",
                "semester": "TEXT",
            },
            "data": [
                (1, 1, 90, "Fall 2020"),
                (1, 2, 85, "Spring 2021"),
                (2, 1, 88, "Fall 2021"),
                (2, 2, 90, "Spring 2022"),
                (3, 3, 92, "Fall 2020"),
                (3, 4, 85, "Spring 2021"),
                (4, 3, 88, "Fall 2021"),
                (4, 4, 86, "Spring 2022"),
                (5, 1, 90, "Fall 2022"),
                (5, 2, 87, "Spring 2023"),
            ],
        },
    }
)


from dbgpt.agent.resource import RDBMSConnectorResource

db_resource = RDBMSConnectorResource("student_manager", connector=connector)


import asyncio
import os

from dbgpt.agent import (
    AgentContext,
    AgentMemory,
    AutoPlanChatManager,
    LLMConfig,
    UserProxyAgent,
)
from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent
from dbgpt.model.proxy import OpenAILLMClient

async def main():
    llm_client = OpenAILLMClient(
        model_alias="deepseek/coder_v2_lite_instruct_16b",  # or other models, eg. "gpt-4o"
        api_base="http://122.227.105.154:32357/v1",
        api_key="XXX",
    )
    context: AgentContext = AgentContext(
        conv_id="test123", language="en", temperature=0.5, max_new_tokens=2048
    )
    agent_memory = AgentMemory()
    agent_memory.gpts_memory.init(conv_id="test123")

    user_proxy = await UserProxyAgent().bind(agent_memory).bind(context).build()

    sql_boy = (
        await DataScientistAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(db_resource)
        .bind(agent_memory)
        .build()
    )
    manager = (
        await AutoPlanChatManager()
        .bind(context)
        .bind(agent_memory)
        .bind(LLMConfig(llm_client=llm_client))
        .build()
    )
    manager.hire([sql_boy])

    await user_proxy.initiate_chat(
        recipient=manager,
        reviewer=user_proxy,
        message="Analyze student scores from at least three dimensions",
    )

    # dbgpt-vis message infos
    print(await agent_memory.gpts_memory.app_link_chat_message("test123"))


if __name__ == "__main__":
    asyncio.run(main())