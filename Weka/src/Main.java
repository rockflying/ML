import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Main {
	private static final String GOOD = "G";
	private static final String BAD = "B";

	private static final String CATEGORY = "category";
	private static final String TEXT = "text";

	private static final int INIT_CAPACITY = 100;

	private static final String[][] TRAINING_DATA = { { "Good", GOOD },
			{ "Wonderful", GOOD }, { "Cool", GOOD }, { "Bad", BAD },
			{ "Disaster", BAD }, { "Terrible", BAD }, {"Not Bad", GOOD}};

	private static final String TEST_DATA = "Sad";

	private static Filter filter = new StringToWordVector();
	private static Classifier classifier = new NaiveBayesMultinomial();

	public static void main(String[] args) throws Exception {
		FastVector categories = new FastVector();
		categories.addElement(GOOD);
		categories.addElement(BAD);

		FastVector attributes = new FastVector();
		attributes.addElement(new Attribute(TEXT, (FastVector) null));
		attributes.addElement(new Attribute(CATEGORY, categories));

		Instances instances = new Instances("Weka", attributes, INIT_CAPACITY);
		instances.setClassIndex(instances.numAttributes() - 1);

		for (String[] pair : TRAINING_DATA) {
			String text = pair[0];
			String category = pair[1];

			Instance instance = createInstanceByText(instances, text);
			instance.setClassValue(category);
			instances.add(instance);
		}

		filter.setInputFormat(instances);
		Instances filteredInstances = Filter.useFilter(instances, filter);
		classifier.buildClassifier(filteredInstances);

		// Test
		String testText = TEST_DATA;
		Instance testInstance = createTestInstance(
				instances.stringFreeStructure(), testText);

		double predicted = classifier.classifyInstance(testInstance);
		String category = instances.classAttribute().value((int) predicted);
		System.out.println(category);
	}

	private static Instance createInstanceByText(Instances data, String text) {
		Attribute textAtt = data.attribute(TEXT);
		int index = textAtt.addStringValue(text);

		Instance instance = new Instance(2);
		instance.setValue(textAtt, index);
		instance.setDataset(data);

		return instance;
	}

	private static Instance createTestInstance(Instances data, String text)
			throws Exception {
		Instance testInstance = createInstanceByText(data, text);
		filter.input(testInstance);
		return filter.output();
	}
}