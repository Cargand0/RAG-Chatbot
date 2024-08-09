from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

template_messages = [
    SystemMessage(content="You are a helpful assistant that extract model number with quantity only without any other text."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

from os.path import expanduser

from langchain_community.llms import LlamaCpp

model_path = expanduser("llama-2-13b-chat.Q3_K_M.gguf")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 30  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  #Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    # streaming=True,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
)

model = Llama2Chat(llm=llm, verbose=True, callback_manager=callback_manager)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory, )

print("Welcome to the chatbot! Type 'exit' to end the conversation.")

while True:
    prompt = input("You: ")
    if prompt.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chain.invoke({"text": prompt})
    # print(response)
    # print("Chatbot:", response)


# print(
#     chain.run(
#         text="What can I see in Vienna? Propose a few locations. Names only, no details."
#     )
# )

# print(chain.run(text="Tell me more about #2."))