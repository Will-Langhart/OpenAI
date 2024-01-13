class WriteForMeGPT:
    def __init__(self):
        self.word_count = 0
        self.sections = []
        self.outline = {}

    def understand_client_needs(self, use, audience, tone, word_count, style, format):
        self.use = use
        self.audience = audience
        self.tone = tone
        self.target_word_count = word_count
        self.style = style
        self.format = format

    def create_outline(self, sections):
        self.sections = sections
        self.outline = {section: {"summary": None, "word_count": 0} for section in sections}

    def manage_word_count(self, section, content):
        word_count = len(content.split())
        self.outline[section]["word_count"] = word_count
        self.word_count += word_count
        return word_count

    def creative_expansion(self, content):
        # Example of a simple content expansion
        expanded_content = content + "\n\n[Additional insightful content here.]"
        return expanded_content

    def sequential_writing(self, section, content):
        if section not in self.sections:
            raise ValueError(f"Section {section} not in outline")
        expanded_content = self.creative_expansion(content)
        self.manage_word_count(section, expanded_content)
        self.outline[section]["summary"] = expanded_content

    def check_content_quality(self, content):
        # Simple quality check example
        if len(content.split()) < 50:
            return "Content quality check: More detail needed."
        return "Content quality check: Good."

    def format_content(self, content):
        if self.format == "markdown":
            formatted_content = f"**{content}**"
        else:
            formatted_content = content
        return formatted_content

    def _format_markdown(self, content):
        # Example markdown formatting
        return f"**{content}**"

    def get_progress_update(self):
        return f"Current word count: {self.word_count} out of {self.target_word_count}"

    def deliver_content(self):
        # Compile all the sections into a single deliverable
        return "\n".join([self.format_content(self.outline[section]["summary"]) for section in self.sections if self.outline[section]["summary"] is not None])

# Example usage:
gpt_writer = WriteForMeGPT()
gpt_writer.understand_client_needs("Blog Post", "General Audience", "Informative", 1000, "Formal", "markdown")
gpt_writer.create_outline(["Introduction", "Body", "Conclusion"])

# Writing content for each section
gpt_writer.sequential_writing("Introduction", "This is the introduction to our topic.")
gpt_writer.sequential_writing("Body", "Detailed discussion on the main subject. Covering various aspects.")
gpt_writer.sequential_writing("Conclusion", "Summarizing the key points and concluding the topic.")

# Print the final content
final_content = gpt_writer.deliver_content()
print(final_content)

# Print progress update
progress = gpt_writer.get_progress_update()
print(progress)

# Content quality check for a section
quality_check = gpt_writer.check_content_quality(gpt_writer.outline["Body"]["summary"])
print(quality_check)
