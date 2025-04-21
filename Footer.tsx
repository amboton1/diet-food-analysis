export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <h2 className="text-xl font-bold">Food-Attention Nexus</h2>
            <p className="text-gray-300">Exploring the relationship between dietary patterns and cognitive performance</p>
          </div>
          <div>
            <p className="text-gray-300">&copy; {new Date().getFullYear()} Food-Attention Nexus Project</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
