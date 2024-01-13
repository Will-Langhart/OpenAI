import requests

class CanvaChatbot:
    def __init__(self):
        self.api_url = "https://chatgpt-plugin.canva.com/generateDesigns"

    def start_conversation(self):
        print("Hello! Excited to bring your visions to life? Start your creative journey with Canva. What will we design together today?")

    def get_design_request(self):
        return input("What message would you like your design to convey? Or, what's the occasion for this design? ")

    def validate_input(self, user_input):
        if not user_input.strip():
            print("Looks like you didn't enter a design idea. Let's try again.")
            return False
        if len(user_input) > 140:
            print("Your input is too long. Let's try to keep it under 140 characters.")
            return False
        return True

    def call_canva_api(self, design_query):
        response = requests.post(self.api_url, json={'query': design_query, 'locale': 'en-US'})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error calling Canva API: {response.status_code}")
            return None

    def display_results(self, results):
        if 'designs' in results:
            self.display_generated_designs(results['designs'])
        elif 'templates' in results:
            self.display_canva_templates(results['templates'])
        else:
            print("No designs or templates found. Let's try a different idea.")

    def display_generated_designs(self, designs):
        print("This technology is new and improving. Please [report these results](https://www.canva.com/help/report-content/) if they don't seem right.")
        if len(designs) == 2:
            self.display_two_designs_side_by_side(designs)
        else:
            self.display_designs_as_list(designs)

    def display_two_designs_side_by_side(self, designs):
        print("| Option 1 | Option 2 |")
        print("|-|-|")
        print(f"| [![]({designs[0]['thumbnail_url']})]({designs[0]['url']}) | [![]({designs[1]['thumbnail_url']})]({designs[1]['url']}) |")

    def display_designs_as_list(self, designs):
        for design in designs:
            print(f"[![]({design['thumbnail_url']})]({design['url']})")

    def display_canva_templates(self, templates):
        for template in templates:
            print(f"[![]({template['thumbnail_url']})]({template['url']})")

    def run(self):
        self.start_conversation()
        while True:
            design_idea = self.get_design_request()
            if self.validate_input(design_idea):
                results = self.call_canva_api(design_idea)
                if results:
                    self.display_results(results)
                    break

if __name__ == "__main__":
    canva_bot = CanvaChatbot()
    canva_bot.run()
