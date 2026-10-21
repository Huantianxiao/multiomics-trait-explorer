"""Browser-independent interaction checks; no live API calls."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class InterfaceTests(unittest.TestCase):
    def test_full_question_analysis_and_setting_invalidation(self):
        app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=90).run()
        self.assertEqual(len(app.exception),0)
        app.chat_input[0].set_value('Find proteins associated with SBP and integrate all analyses.').run()
        self.assertEqual(len(app.exception),0)
        next(b for b in app.button if b.label=='Run this analysis').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        result=app.session_state['output']
        self.assertGreater(result['summary']['protein_hits'],0)
        self.assertIn('coloc',result['tables'])
        self.assertIn('go',result['tables'])
        self.assertEqual(len(app.tabs),6)
        app.sidebar.multiselect[0].unselect('BMI').run()
        self.assertNotIn('output',app.session_state)
        self.assertNotIn('pending',app.session_state)

    def test_example_button_and_invalid_request_clear_results(self):
        app=AppTest.from_file(str(Path(__file__).with_name('app.py')),default_timeout=90).run()
        next(b for b in app.button if b.label=='Explore HbA1c · all modules').click().run()
        self.assertEqual(app.session_state['pending']['trait'],'HbA1c')
        next(b for b in app.button if b.label=='Run this analysis').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        self.assertIn('output',app.session_state)
        app.chat_input[0].set_value('Find proteins associated with mortality.').run()
        self.assertEqual(len(app.error),1)
        self.assertNotIn('output',app.session_state)
        self.assertNotIn('pending',app.session_state)


if __name__=='__main__':
    unittest.main(verbosity=2)
