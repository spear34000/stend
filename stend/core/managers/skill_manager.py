import os
import importlib.util
from typing import List

class SkillManager:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills = []

    def load_skills(self) -> List[str]:
        self.skills = []
        loaded_names = []
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)
            return []

        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                skill_name = filename[:-3]
                file_path = os.path.join(self.skills_dir, filename)
                
                spec = importlib.util.spec_from_file_location(skill_name, file_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                
                self.skills.append(mod)
                loaded_names.append(skill_name)
        
        return loaded_names

    def dispatch(self, event_type: str, data: dict):
        for skill in self.skills:
            try:
                if event_type == "message" and hasattr(skill, "on_message"):
                    skill.on_message(data)
                elif event_type == "stend_event" and hasattr(skill, "on_stend_event"):
                    skill.on_stend_event(data)
            except Exception as e:
                print(f"[SkillManager] Error in skill {getattr(skill, '__name__', 'unknown')}: {e}")
