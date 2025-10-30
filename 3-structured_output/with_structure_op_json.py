from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

google_api_key = os.getenv('Gemini_api_key')
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',
                               api_key=google_api_key)

json_schema  = {
    'title': 'Review',
    'type':'object',
    'properties':{
        "Key_theme":{
            'type':'array',
            'items':{
                'type':'string'
            },
            'description':'Tell major review is about'
        },
        'Summary':{
            'type':'string',
            'description':'A breif summary with review'
        },
        'Sentiment':{
            'type':'string',
            'enum': ['pos','neg'],
            'description':'Tell sentiments is neutral,positive,negative'
        },
        'pros':{
            'type':['array','null'],
            'items':{
                'type':'string'
            },
            'description':'write down all the pros'
        },
        'con':{
            'type':['array','null'],
            'items':{
                'type':'string'
            }
        }

    }
}

structured_model = model.with_structured_output(json_schema)
temp = PromptTemplate(
    template='You are a good feedback reviewer who reviews the feedback so please review the given feedback {feedback}',
    input_variables=['feedback']
)
chain = temp | structured_model

Result = chain.invoke('''The performance is very good at the price range of ₹24k-₹26k. The phone comes with dual speaker so sound quality is very good. It has glass back with metal frame so the phone may feel uncomfortable at first but with time it is good in handling. The speed is good. Durability is better and the phone doesn't hang too much. The phone is verall value for money.
''')
# structured_model = model.with_structured_output(Review.__annotations__)

print(Result)
