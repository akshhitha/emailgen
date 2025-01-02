import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature =0, groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama-3.1-70b-versatile")
    def extract_jobs(self, page_data):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION: 
        The scraped text is from the careers page of a website.
        Your have to extract the job posting information and return in JSON format including the keys: 'role', 'experience', 'skills', 'description'.
        Only return the JSON format.
        ### VALID JSON (NO PREAMBLE):    
        """
    )
        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input={'page_data':page_data})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
         """ 
        ### JOB DESCRIPTION:
        {job_description}

        ###INSTRUCTION:
        You must write a cold email to the client based on the above description. Add the relevant link from the following to show highest relevancy: {link_list} express your interest for the job
        """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list":links})
        return res.content
    