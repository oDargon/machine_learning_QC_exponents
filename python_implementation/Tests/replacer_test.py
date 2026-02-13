from exponent_handler import *
from molcas_handler import *
from pathlib import Path

current_dir  = Path.cwd()
work_dir     = current_dir / "Work"
template_dir = current_dir / "Templates/Be_template_1.inp"
# print(work_dir)



exp = Exponent_Set.from_file("Resources/Be_HF_1.expo")
# print(exp)

exp.print_exponents()

# job1 = Molcas_Job( "job_1", work_dir, template_dir, exp, input_name="Cool" )