import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div>
      {/* Hero Section */}
      <div className="hero-section flex items-center justify-center">
        <div className="text-center text-white p-4">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Food-Attention Nexus</h1>
          <p className="text-xl md:text-2xl mb-8">
            Exploring the relationship between dietary patterns and cognitive performance
          </p>
          <Link 
            href="/overview" 
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition duration-300"
          >
            Explore the Project
          </Link>
        </div>
      </div>

      {/* Introduction Section */}
      <section className="py-16 px-4 max-w-7xl mx-auto">
        <h2 className="section-heading mx-auto">Project Introduction</h2>
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div>
            <p className="text-lg mb-4">
              The Food-Attention Nexus project investigates the complex relationship between dietary patterns and cognitive performance, specifically focusing on attention span and cognitive function.
            </p>
            <p className="text-lg mb-4">
              Using advanced data science and machine learning techniques, we've analyzed how different dietary patterns affect cognitive metrics, identified which populations are most sensitive to these effects, and developed predictive models for personalized nutrition recommendations.
            </p>
            <p className="text-lg">
              Our findings provide actionable insights for improving cognitive function through targeted dietary interventions at public health, healthcare provider, and individual levels.
            </p>
          </div>
          <div className="flex justify-center">
            <Image 
              src="/images/correlation_matrix.png" 
              alt="Correlation Matrix" 
              width={500} 
              height={400}
              className="rounded-lg shadow-lg"
            />
          </div>
        </div>
      </section>

      {/* Key Findings Preview */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <h2 className="section-heading mx-auto">Key Findings</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="key-finding">
              <h3>Mediterranean Diet Benefits</h3>
              <p>
                Our analysis consistently identified the Mediterranean diet pattern as having the strongest positive association with cognitive performance across all attention metrics.
              </p>
            </div>
            <div className="key-finding">
              <h3>Fiber Intake is Critical</h3>
              <p>
                Among all macronutrients analyzed, fiber intake emerged as the strongest predictor of cognitive performance, with high-fiber diets correlating with improved sustained attention.
              </p>
            </div>
            <div className="key-finding">
              <h3>Processed Food Reduction</h3>
              <p>
                Each additional daily serving of processed food was associated with measurable decreases in attention metrics, with the western diet pattern showing the strongest negative correlation.
              </p>
            </div>
          </div>
          <div className="text-center mt-8">
            <Link 
              href="/findings" 
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
            >
              View All Findings
            </Link>
          </div>
        </div>
      </section>

      {/* Visualizations Preview */}
      <section className="py-16 px-4 max-w-7xl mx-auto">
        <h2 className="section-heading mx-auto">Visualizations</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="visualization-card">
            <Image 
              src="/images/top_correlations_reaction_time_ms.png" 
              alt="Top Correlations with Reaction Time" 
              width={400} 
              height={300}
              className="w-full h-48 object-cover"
            />
            <div className="p-4">
              <h3 className="font-bold text-lg mb-2">Dietary Impacts on Reaction Time</h3>
              <p className="text-gray-700">
                Visualization of the top dietary factors correlated with reaction time performance.
              </p>
            </div>
          </div>
          <div className="visualization-card">
            <Image 
              src="/images/top_correlations_sustained_attention_score.png" 
              alt="Top Correlations with Sustained Attention" 
              width={400} 
              height={300}
              className="w-full h-48 object-cover"
            />
            <div className="p-4">
              <h3 className="font-bold text-lg mb-2">Factors Affecting Sustained Attention</h3>
              <p className="text-gray-700">
                Analysis of dietary and lifestyle factors that influence sustained attention capacity.
              </p>
            </div>
          </div>
          <div className="visualization-card">
            <Image 
              src="/images/feature_category_importance.png" 
              alt="Feature Category Importance" 
              width={400} 
              height={300}
              className="w-full h-48 object-cover"
            />
            <div className="p-4">
              <h3 className="font-bold text-lg mb-2">Feature Category Importance</h3>
              <p className="text-gray-700">
                Relative importance of different feature categories in predicting cognitive performance.
              </p>
            </div>
          </div>
        </div>
        <div className="text-center mt-8">
          <Link 
            href="/visualizations" 
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
          >
            Explore All Visualizations
          </Link>
        </div>
      </section>

      {/* Recommendations Preview */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <h2 className="section-heading mx-auto">Recommendations</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="recommendation">
              <h3>Public Health Initiatives</h3>
              <p>
                Develop evidence-based messaging around Mediterranean diet principles for cognitive health and create educational campaigns highlighting the cognitive benefits of fiber and risks of processed foods.
              </p>
            </div>
            <div className="recommendation">
              <h3>Healthcare Provider Strategies</h3>
              <p>
                Create screening tools to identify individuals most likely to benefit from dietary interventions for cognitive health and implement personalized nutrition algorithms in clinical settings.
              </p>
            </div>
          </div>
          <div className="text-center mt-8">
            <Link 
              href="/recommendations" 
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
            >
              View All Recommendations
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
